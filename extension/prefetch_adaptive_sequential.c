#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <time.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

/* 这些是项目自定义的头文件或 eBPF 编译后自动生成的骨架文件 */
#include "prefetch_adaptive_sequential.skel.h" // eBPF 程序的 C 语言骨架 (Skeleton)
#include "cleanup_struct_ops.h"              // 用于清理残留的 eBPF 挂载点
#include "nvml_monitor.h"                    // 封装了 NVIDIA NVML 库，用于读取 GPU 状态

/* libbpf 的日志回调函数，将内核 eBPF 加载过程中的调试信息输出到标准错误(stderr) */
static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

/* 预取方向常量定义，必须与内核 eBPF 代码中的定义保持一致 */
#define PREFETCH_FORWARD       0 // 向前预取：触发缺页异常的地址之后的页面
#define PREFETCH_BACKWARD      1 // 向后预取：触发缺页异常的地址之前的页面
#define PREFETCH_FORWARD_START 2 // 从区域起始位置向前预取

/* 全局标志位，用于捕捉 Ctrl+C 信号以实现优雅退出 */
static volatile bool exiting = false;
/* NVML 设备句柄，用于后续查询 GPU 状态 */
static nvmlDevice_t nvml_device = NULL;

/* * 配置结构体 (Configuration)
 * 保存从命令行参数解析出的用户设定
 */
static struct {
    int fixed_pct;           /* 固定预取百分比。-1 表示使用自适应模式 */
    unsigned int min_pct;    /* 自适应模式下的最小预取百分比 */
    unsigned int max_pct;    /* 自适应模式下的最大预取百分比 */
    unsigned long long max_mbps;  /* 标定的最大 PCIe 带宽 (MB/s)，用于计算当前负载比例 */
    int invert;              /* 是否反转自适应逻辑 */
    unsigned int direction;  /* 预取方向：0=forward, 1=backward */
    unsigned int num_pages;  /* 固定的预取页数。如果 >0，则忽略百分比逻辑 */
} config = {
    .fixed_pct = -1,         /* 默认开启自适应 */
    .min_pct = 30,
    .max_pct = 100,
    .max_mbps = 20480ULL,    /* 默认最大带宽为 20 GB/s */
    .invert = 0,
    .direction = PREFETCH_FORWARD,  /* 默认向前预取 */
    .num_pages = 0,          /* 默认不固定页数，走百分比逻辑 */
};

/* 信号处理函数：当用户按下 Ctrl+C (SIGINT) 或发送 SIGTERM 时，触发退出循环 */
void handle_signal(int sig) {
    exiting = true;
}

/* * 核心功能：使用 NVML 获取 GPU 当前的 PCIe 吞吐量 (MB/s) 
 */
static unsigned long long get_pcie_throughput_mbps(void) {
    if (!nvml_device) {
        return 0; // 如果 NVML 没初始化成功，返回 0
    }

    // 调用自定义 nvml_monitor.h 中的函数读取 KB/s，然后转换为 MB/s
    unsigned long long throughput_kbps = nvml_get_pcie_throughput_kbps(nvml_device);
    return throughput_kbps / 1024;
}

/* * 核心逻辑：根据当前的 PCIe 带宽吞吐量，计算应该下发给内核的预取百分比。
 *
 * 正常模式 (invert=0): 正相关
 * 流量高 -> 说明程序在大量拷贝数据 -> 提高预取比例 -> 激进预取
 * 流量低 -> 说明程序不需要大量数据 -> 降低预取比例 -> 保守预取
 *
 * 反转模式 (invert=1): 负相关
 * 流量高 -> PCIe 带宽拥挤 -> 降低预取比例 -> 防止无用预取阻塞正常访存 (带宽受限场景)
 * 流量低 -> PCIe 带宽空闲 -> 提高预取比例 -> 利用空闲带宽提前搬运数据 (带宽充裕场景)
 */
static unsigned int calculate_prefetch_percentage(unsigned long long throughput_mbps) {
    // 如果当前吞吐量已经超过了我们设定的最大阈值，直接返回极值
    if (throughput_mbps >= config.max_mbps) {
        return config.invert ? config.min_pct : config.max_pct;
    }

    // 计算当前吞吐量占最大吞吐量的比例 (0.0 ~ 1.0)
    double ratio = (double)throughput_mbps / (double)config.max_mbps; 

    // 如果开启了反转，则将比例翻转 (1.0 变 0.0)
    if (config.invert) {
        ratio = 1.0 - ratio;  
    }

    // 根据比例在 [min_pct, max_pct] 之间线性插值
    // + 0.5 是为了实现四舍五入
    unsigned int pct = (unsigned int)(config.min_pct + 
                                      (config.max_pct - config.min_pct) * ratio + 0.5);
    
    // 边界安全检查，确保输出不越界
    if (pct < config.min_pct) pct = config.min_pct;
    if (pct > config.max_pct) pct = config.max_pct;
    return pct;
}

/* 打印命令行使用帮助说明 */
static void print_usage(const char *prog) {
    // ... [打印帮助信息的代码，此处注释略过，代码本身很直观] ...
    printf("Usage: %s [OPTIONS]\n", prog);
    // ... 
}

int main(int argc, char **argv) {
    struct prefetch_adaptive_sequential_bpf *skel; // eBPF 骨架对象
    struct bpf_link *link = NULL;                  // eBPF 挂载链接
    int err;
    int verify_err;
    int pct_map_fd;                                // 指向内核 eBPF map 的文件描述符 (用于传递百分比)
    unsigned int key = 0;                          // eBPF map 的键（这里我们通常只用 key=0 存单个值）
    int opt;

    // --- 阶段 1：解析命令行参数 ---
    while ((opt = getopt(argc, argv, "p:m:M:b:id:n:h")) != -1) {
        switch (opt) {
        case 'p': // 固定百分比
            config.fixed_pct = atoi(optarg);
            if (config.fixed_pct < 0 || config.fixed_pct > 100) {
                fprintf(stderr, "Error: percentage must be 0-100\n");
                return 1;
            }
            break;
        case 'm': // 自适应最小值
            config.min_pct = (unsigned int)atoi(optarg);
            if (config.min_pct > 100) return 1;
            break;
        case 'M': // 自适应最大值
            config.max_pct = (unsigned int)atoi(optarg);
            if (config.max_pct > 100) return 1;
            break;
        case 'b': // 最大 PCIe 带宽
            config.max_mbps = (unsigned long long)atoll(optarg);
            break;
        case 'i': // 反转逻辑
            config.invert = 1;
            break;
        case 'd': // 预取方向
            if (strcmp(optarg, "forward") == 0 || strcmp(optarg, "f") == 0) {
                config.direction = PREFETCH_FORWARD;
            } else if (strcmp(optarg, "backward") == 0 || strcmp(optarg, "b") == 0) {
                config.direction = PREFETCH_BACKWARD;
            } else if (strcmp(optarg, "forward_start") == 0 || strcmp(optarg, "fs") == 0) {
                config.direction = PREFETCH_FORWARD_START;
            } else {
                return 1; // 参数错误
            }
            break;
        case 'n': // 固定页数
            config.num_pages = (unsigned int)atoi(optarg);
            break;
        case 'h': // 帮助
            print_usage(argv[0]);
            return 0;
        default:
            return 1;
        }
    }

    // 校验自适应上下限是否合法
    if (config.min_pct > config.max_pct) {
        fprintf(stderr, "Error: min percentage (%u) cannot be greater than max (%u)\n",
                config.min_pct, config.max_pct);
        return 1;
    }

    // 注册信号处理，捕获 Ctrl+C 以便在退出时清理 eBPF 挂载
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    // 设置 libbpf 打印回调函数
    libbpf_set_print(libbpf_print_fn);

    // --- 阶段 2：环境初始化 ---
    
    // 如果没有使用固定比例（即启用了自适应），则初始化 NVML 准备读取 GPU 数据
    if (config.fixed_pct < 0) {
        nvml_device = nvml_init_device();
        if (!nvml_device) {
            fprintf(stderr, "Warning: Failed to initialize NVML, using fixed 100%% mode\n");
            // 如果 NVML 初始化失败，降级为固定 100% 模式
            config.fixed_pct = 100;
        }
    }

    // 清理系统中可能遗留的旧 struct_ops 挂载
    // struct_ops 是内核提供的一种机制，允许 eBPF 程序替换内核中特定的函数指针表（例如这里的 UVM 策略调度器）
    err = cleanup_old_struct_ops();
    if (err == -EEXIST) {
        fprintf(stderr,
                "Refusing to load adaptive_sequential: stale UVM struct_ops instances are still present.\n");
        return 1;
    }

    err = verify_no_uvm_struct_ops_instances("startup");
    if (err) {
        fprintf(stderr,
                "Refusing to load adaptive_sequential: unable to verify a clean UVM struct_ops state.\n");
        return 1;
    }

    // --- 阶段 3：加载 eBPF 程序到内核 ---
    
    // 打开 BPF 骨架（只读取并解析 ELF 文件，还没真正注入内核）
    skel = prefetch_adaptive_sequential_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    // 真正将 BPF 程序和 Maps 加载(Load)到 Linux 内核中，并在内核执行验证器(Verifier)校验
    err = prefetch_adaptive_sequential_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup; // 失败则跳到末尾执行清理
    }

    // --- 阶段 4：配置内核侧的 eBPF Maps 参数 ---
    // eBPF Maps 是用户态和内核态 eBPF 程序共享数据的"桥梁"

    // 1. 获取并设置"预取百分比" (prefetch_pct_map)
    pct_map_fd = bpf_map__fd(skel->maps.prefetch_pct_map);
    unsigned int initial_pct = (config.fixed_pct >= 0) ? (unsigned int)config.fixed_pct : config.max_pct;
    err = bpf_map_update_elem(pct_map_fd, &key, &initial_pct, BPF_ANY); // 将用户态的值写入内核 map

    // 2. 获取并设置"预取方向" (prefetch_direction_map)
    int dir_map_fd = bpf_map__fd(skel->maps.prefetch_direction_map);
    err = bpf_map_update_elem(dir_map_fd, &key, &config.direction, BPF_ANY);

    // 3. 获取并设置"固定页数" (prefetch_num_pages_map)
    int num_map_fd = bpf_map__fd(skel->maps.prefetch_num_pages_map);
    err = bpf_map_update_elem(num_map_fd, &key, &config.num_pages, BPF_ANY);

    // --- 阶段 5：挂载(Attach) eBPF 程序 ---
    // 将写好的 eBPF 函数表正式挂载到内核的 UVM 调度子系统中
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_adaptive_sequential);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Successfully loaded and attached BPF adaptive_sequential policy!\n");
    // ... [打印当前配置信息，省略注释] ...

    // --- 阶段 6：主循环 (守护进程) ---
    // 程序留在这里一直运行，保持 eBPF 的挂载状态，并动态更新策略
    while (!exiting) {
        if (config.fixed_pct < 0) {
            /* 自适应模式：每秒轮询一次带宽，动态更新内核参数 */
            unsigned long long throughput = get_pcie_throughput_mbps();
            unsigned int pct = calculate_prefetch_percentage(throughput);

            // 动态将计算出的新百分比通过 map 写入内核，内核中的 eBPF 程序下一次触发时就会使用新参数
            err = bpf_map_update_elem(pct_map_fd, &key, &pct, BPF_ANY);
            if (err) {
                fprintf(stderr, "Failed to update prefetch percentage: %d\n", err);
            } else {
                printf("[%ld] PCIe: %llu MB/s -> Prefetch: %u%%\n",
                       time(NULL), throughput, pct);
            }
        }
        sleep(1); // 睡眠 1 秒，降低轮询带来的 CPU 占用
    }

    // --- 阶段 7：清理与退出 ---
    // 当收到 Ctrl+C 信号跳出循环后，来到这里
    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link); // 销毁挂载链接，内核将恢复默认的 UVM 策略
    link = NULL;

    verify_err = verify_no_uvm_struct_ops_instances("shutdown");
    if (!err && verify_err)
        err = verify_err;

cleanup:
    if (link)
        bpf_link__destroy(link);
    prefetch_adaptive_sequential_bpf__destroy(skel); // 释放骨架内存

    if (nvml_device) {
        nvml_cleanup(); // 关闭 NVML 库
    }

    return err < 0 ? -err : 0;
}
