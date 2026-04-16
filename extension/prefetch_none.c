#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>        // libbpf核心API
#include <bpf/libbpf.h>     // libbpf高级API

#include "prefetch_none.skel.h" // 自动生成的BPF骨架头文件 (对应prefetch_none.bpf.c)
#include "cleanup_struct_ops.h" // 清理旧实例的辅助函数

/**
 * @brief libbpf的打印回调函数
 * 
 * 此函数用于接收libbpf库内部的日志信息，并将其输出到标准错误流。
 * 这对于调试BPF程序加载和运行过程中的问题非常有用。
 */
static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

// 全局变量，用于控制主循环的退出
static volatile bool exiting = false;

/**
 * @brief 信号处理器
 * 
 * 当进程收到SIGINT(Ctrl+C)或SIGTERM信号时，该函数会被调用，
 * 将全局的`exiting`标志设置为true，从而让主循环安全退出。
 */
void handle_signal(int sig) {
    exiting = true;
}

int main(int argc, char **argv) {
    struct prefetch_none_bpf *skel; // BPF骨架结构体，是与内核BPF程序交互的主要句柄
    struct bpf_link *link;          // BPF链接对象，用于管理struct_ops的挂载
    int err;                        // 通用错误码

    // 注册信号处理器，以便能够优雅地关闭程序
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* 设置libbpf的调试输出 */
    libbpf_set_print(libbpf_print_fn);

    /* 
     * 在加载新的struct_ops实例之前，检查并清理系统中可能存在的
     * 旧的、未正确卸载的同类型实例。这是一个重要的清理步骤，
     * 以避免冲突和资源泄漏。
     */
    cleanup_old_struct_ops();

    /* 
     * 打开BPF应用程序骨架。
     * 这会加载BPF字节码并准备好用户态的控制接口，但尚未将其加载到内核。
     */
    skel = prefetch_none_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    /* 
     * 将BPF程序正式加载到内核。
     * 这一步会验证BPF代码的安全性并分配内核资源。
     */
    err = prefetch_none_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        // 加载失败，跳转到清理部分
        goto cleanup;
    }

    /* 
     * 激活并挂载struct_ops。
     * 这是关键一步，它将用户态定义的`uvm_ops_none`结构体（包含eBPF程序指针）
     * 传递给内核，使内核开始调用这些eBPF程序来处理GPU内存预取相关的事件。
     */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_none);
    if (!link) {
        // 获取具体的错误码
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    // 向用户报告程序已成功加载并运行
    printf("Successfully loaded and attached BPF prefetch_none policy!\n");
    printf("The prefetch policy is now active and will DISABLE all prefetching.\n");
    printf("Monitor tracepipe for BPF debug output.\n");
    printf("\nPress Ctrl-C to exit and detach the policy...\n");

    /* 
     * 主循环：保持程序运行，等待退出信号。
     * 在此期间，内核会根据内存访问模式调用已挂载的eBPF程序来决定预取行为。
     * 由于策略是禁用预取，所有预取请求都将被忽略。
     */
    while (!exiting) {
        sleep(1);
    }

    // 用户发出退出信号，开始执行清理程序
    printf("\nDetaching struct_ops...\n");
    // 销毁链接对象，这会通知内核停止调用我们的eBPF程序
    bpf_link__destroy(link);

cleanup:
    // 销毁BPF骨架，释放所有相关的用户态和内核态资源
    prefetch_none_bpf__destroy(skel);
    // 返回错误码，如果err为负则取其绝对值作为返回码
    return err < 0 ? -err : 0;
}