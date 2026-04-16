#include <linux/init.h>        // 包含模块初始化所需的宏和函数 (如module_init, module_exit)
#include <linux/module.h>      // 包含定义内核模块所需的核心宏 (如MODULE_LICENSE, MODULE_AUTHOR)
#include <linux/kernel.h>      // 包含内核打印函数 (如printk) 和基本类型
#include <linux/bpf.h>         // 包含BPF框架的核心定义
#include <linux/btf.h>         // 包含BTF (BPF Type Format) 相关定义，用于类型信息
#include <linux/btf_ids.h>     // 包含BTF IDs相关定义，用于标识kfuncs
#include <linux/proc_fs.h>     // 包含创建proc文件系统的API
#include <linux/seq_file.h>    // 包含创建序列化文件的API (通常与proc配合使用)
#include <linux/bpf_verifier.h>// 包含BPF验证器相关的API和结构体
#include "uvm_bpf_struct_ops.h"// 包含用户自定义的BPF struct_ops相关类型和动作码
#include "uvm_migrate.h"       // 包含GPU内存迁移相关的函数声明

/* 
 * 为了兼容较低版本的内核，定义一些在新版内核中才引入的宏。
 * BTF (BPF Type Format) 用于内核和BPF程序之间的类型信息交换。
 */
#ifndef BTF_SET8_KFUNCS
/* 此标志表示BTF_SET8结构持有一个或多个kfunc (kernel function) */
#define BTF_SET8_KFUNCS		(1 << 0)
#endif

#ifndef BTF_KFUNCS_START
// 定义一个用于收集kfunc ID的静态结构体，用于批量注册
#define BTF_KFUNCS_START(name) static struct btf_id_set8 __maybe_unused name = { .flags = BTF_SET8_KFUNCS };
#endif

#ifndef BTF_KFUNCS_END
// 结束kfunc ID集合的定义
#define BTF_KFUNCS_END(name)
#endif

/*
 * 定义一个名为`gpu_mem_ops`的结构体，它充当了内核代码和BPF程序之间的桥梁。
 * 结构体中的每个函数指针都是一个“钩子”，内核在特定事件发生时会调用这些指针指向的BPF程序。
 * 这个结构体的定义必须与用户态加载BPF程序时所引用的结构体（例如在BPF源码中通过SEC(".struct_ops")定义的）完全一致。
 */
struct gpu_mem_ops {
	// 一个测试触发器，用于从用户态（如通过proc文件）触发BPF程序执行
	int (*gpu_test_trigger)(const char *buf, int len);

	/* 预取 (Prefetch) 相关的钩子 */
	// 主预取决策点：当内核认为需要预取时调用
	int (*gpu_page_prefetch)(
		uvm_page_index_t page_index,                           // 触发预取的页面索引
		uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,          // 用于分析访问模式的位图树
		uvm_va_block_region_t *max_prefetch_region,           // 内核建议的最大预取区域
		uvm_va_block_region_t *result_region);                // [out] BPF程序输出的最终预取区域

	// 预取迭代钩子：在内核遍历预取位图树时被调用
	int (*gpu_page_prefetch_iter)(
		uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
		uvm_va_block_region_t *max_prefetch_region,
		uvm_va_block_region_t *current_region,
		unsigned int counter,
		uvm_va_block_region_t *prefetch_region);

	/* 内存块置换 (Eviction) 相关的钩子 */
	// 块激活钩子：当一个内存块变为可驱逐状态时调用
	int (*gpu_block_activate)(
		uvm_pmm_gpu_t *pmm,                    // GPU物理内存管理器
		uvm_gpu_chunk_t *chunk,                // 被激活的GPU内存块
		struct list_head *list);               // 内存块所属的链表

	// 块访问钩子：当一个内存块被访问时调用
	int (*gpu_block_access)(
		uvm_pmm_gpu_t *pmm,                    // GPU物理内存管理器
		uvm_gpu_chunk_t *chunk,                // 被访问的GPU内存块
		struct list_head *list);               // 内存块所属的链表

	// 驱逐准备钩子：在实际开始驱逐前调用
	int (*gpu_evict_prepare)(
		uvm_pmm_gpu_t *pmm,                    // GPU物理内存管理器
		struct list_head *va_block_used,         // 已使用的虚拟地址块列表
		struct list_head *va_block_unused);      // 未使用的虚拟地址块列表
};

/* 
 * 定义一个全局的`gpu_mem_ops`实例指针，用于存储从用户态加载并注册进来的BPF程序集合。
 * `__rcu`标记表示该指针受RCU (Read-Copy-Update) 机制保护，确保在多线程环境下的安全读取。
 */
static struct gpu_mem_ops __rcu *uvm_ops;

/* 用于触发BPF程序执行的proc文件句柄 */
static struct proc_dir_entry *trigger_file;

/*
 * CFI (Control Flow Integrity) Stub Functions (CFI存根函数)。
 * 这些是内核在编译时需要的占位符函数，以确保类型安全和控制流完整性。
 * 它们不是实际被调用的函数，而是代表了`gpu_mem_ops`结构体中函数指针的原型。
 */
static int gpu_mem_ops__gpu_test_trigger(const char *buf, int len) { return 0; }
static int gpu_mem_ops__gpu_page_prefetch(
	uvm_page_index_t page_index,
	uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	uvm_va_block_region_t *max_prefetch_region,
	uvm_va_block_region_t *result_region)
{ 
	return UVM_BPF_ACTION_DEFAULT; 
}
static int gpu_mem_ops__gpu_page_prefetch_iter(
	uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	uvm_va_block_region_t *max_prefetch_region,
	uvm_va_block_region_t *current_region,
	unsigned int counter,
	uvm_va_block_region_t *prefetch_region)
{ 
	return UVM_BPF_ACTION_DEFAULT; 
}
static int gpu_mem_ops__gpu_block_activate(
	uvm_pmm_gpu_t *pmm,
	uvm_gpu_chunk_t *chunk,
	struct list_head *list)
{ 
	return UVM_BPF_ACTION_DEFAULT; 
}
static int gpu_mem_ops__gpu_block_access(
	uvm_pmm_gpu_t *pmm,
	uvm_gpu_chunk_t *chunk,
	struct list_head *list)
{ 
	return UVM_BPF_ACTION_DEFAULT; 
}
static int gpu_mem_ops__gpu_evict_prepare(
	uvm_pmm_gpu_t *pmm,
	struct list_head *va_block_used,
	struct list_head *va_block_unused)
{ 
	return UVM_BPF_ACTION_DEFAULT; 
}

/* 将这些存根函数组合成一个结构体实例，供BPF框架内部使用 */
static struct gpu_mem_ops __bpf_ops_gpu_mem_ops = {
	.gpu_test_trigger = gpu_mem_ops__gpu_test_trigger,
	.gpu_page_prefetch = gpu_mem_ops__gpu_page_prefetch,
	.gpu_page_prefetch_iter = gpu_mem_ops__gpu_page_prefetch_iter,
	.gpu_block_activate = gpu_mem_ops__gpu_block_activate,
	.gpu_block_access = gpu_mem_ops__gpu_block_access,
	.gpu_evict_prepare = gpu_mem_ops__gpu_evict_prepare,
};

/* 开始定义kfunc (kernel function) */
__bpf_kfunc_start_defs();

/**
 * @brief 一个示例性的kfunc，用于从BPF程序中调用内核函数。
 * @note 这里仅为测试目的，实际功能未实现。
 */
__bpf_kfunc int bpf_gpu_strstr(const char *str, u32 str__sz, const char *substr, u32 substr__sz)
{
	return -1;
}

/**
 * @brief kfunc: 允许BPF程序修改预取区域。
 * 这是BPF程序与内核交互的重要接口，可以用来动态设置预取范围。
 */
__bpf_kfunc void bpf_gpu_set_prefetch_region(uvm_va_block_region_t *region,
					     uvm_page_index_t first,
					     uvm_page_index_t outer)
{
	if (!region)                                                                                                 
		return;
	region->first = first; // 设置区域起始页索引
	region->outer = outer; // 设置区域结束页索引
}

/**
 * @brief kfunc: 将GPU内存块移动到链表头部。
 * 这常用于实现LRU等置换策略，将最近使用的块放到最前面。
 */
__bpf_kfunc void bpf_gpu_block_move_head(uvm_gpu_chunk_t *chunk,
					     struct list_head *list)
{
	if (!chunk || !list)
		return;

	/* 移动前先检查块是否已经在某个链表中 */
	if (list_empty(&chunk->list))
		return;

	list_move(&chunk->list, list);
}

/**
 * @brief kfunc: 将GPU内存块移动到链表尾部。
 * 这常用于实现FIFO等置换策略，将最早放入的块放在最后面（最容易被驱逐）。
 */
__bpf_kfunc void bpf_gpu_block_move_tail(uvm_gpu_chunk_t *chunk,
					     struct list_head *list)
{
	if (!chunk || !list)
		return;

	/* 移动前先检查块是否已经在某个链表中 */
	if (list_empty(&chunk->list))
		return;

	list_move_tail(&chunk->list, list);
}

/* ===== Cross-block prefetch kfunc ===== */

/**
 * @brief kfunc: 将一段虚拟地址范围迁移到GPU内存。
 * 这是一个可睡眠的kfunc (KF_SLEEPABLE)，意味着它可以在可能引起进程调度的上下文中被调用。
 * 它依赖于`uvm_migrate_bpf`函数来完成实际的迁移工作。
 */
__bpf_kfunc int bpf_gpu_migrate_range(u64 va_space_handle, u64 addr, u64 length)
{
	// 将句柄转换回内核指针
	uvm_va_space_t *va_space = (uvm_va_space_t *)va_space_handle;
	if (!va_space || !length)
		return -EINVAL;
	// 调用内核迁移函数
	return (int)uvm_migrate_bpf(va_space, addr, length);
}

/* 结束kfunc定义 */
__bpf_kfunc_end_defs();

/* 
 * 定义一个BTF ID集合，将上面声明的所有kfunc函数ID注册到内核。
 * 这样BPF程序才能在编译时找到并链接到这些内核函数。
 */
BTF_KFUNCS_START(uvm_bpf_kfunc_ids_set)
BTF_ID_FLAGS(func, bpf_gpu_strstr) // 注册函数ID
BTF_ID_FLAGS(func, bpf_gpu_set_prefetch_region, KF_TRUSTED_ARGS) // 标记为可信参数
BTF_ID_FLAGS(func, bpf_gpu_block_move_head, KF_TRUSTED_ARGS)
BTF_ID_FLAGS(func, bpf_gpu_block_move_tail, KF_TRUSTED_ARGS)
BTF_ID_FLAGS(func, bpf_gpu_migrate_range, KF_SLEEPABLE) // 标记为可睡眠
BTF_KFUNCS_END(uvm_bpf_kfunc_ids_set)

/* 注册这个kfunc ID集合 */
static const struct btf_kfunc_id_set uvm_bpf_kfunc_set = {
	.owner = THIS_MODULE, // 模块所有者
	.set = &uvm_bpf_kfunc_ids_set, // 指向ID集合
};

/* --- struct_ops 验证器和回调函数 --- */

/* struct_ops初始化回调，在struct_ops注册时调用 */
static int gpu_mem_ops_init(struct btf *btf) { return 0; }

/* 检查BPF程序是否有权限访问`gpu_mem_ops`结构体的成员 */
static bool gpu_mem_ops_is_valid_access(int off, int size,
					    enum bpf_access_type type,
					    const struct bpf_prog *prog,
					    struct bpf_insn_access_aux *info)
{
	// 使用BTF进行基于类型的上下文访问检查
	return bpf_tracing_btf_ctx_access(off, size, type, prog, info);
}

/* 允许BPF程序使用特定的辅助函数 */
static const struct bpf_func_proto *
gpu_mem_ops_get_func_proto(enum bpf_func_id func_id,
			       const struct bpf_prog *prog)
{
	// 返回基础的辅助函数原型，如trace_printk等
	return bpf_base_func_proto(func_id, prog);
}

/* struct_ops的验证器操作集，定义了如何验证挂载到此struct_ops上的BPF程序 */
static const struct bpf_verifier_ops gpu_mem_ops_verifier_ops = {
	.is_valid_access = gpu_mem_ops_is_valid_access, // 访问权限检查
	.get_func_proto = gpu_mem_ops_get_func_proto,   // 允许的辅助函数列表
};

/* 初始化`gpu_mem_ops`结构体成员的回调 */
static int gpu_mem_ops_init_member(const struct btf_type *t,
				       const struct btf_member *member,
				       void *kdata, const void *udata)
{
	return 0; // 无需特殊初始化
}

/* struct_ops注册回调：当用户态程序加载BPF程序并挂载时被调用 */
static int gpu_mem_ops_reg(void *kdata, struct bpf_link *link)
{
	struct gpu_mem_ops *ops = kdata;

	// 使用原子比较和交换指令，确保同时只有一个实例被注册
	if (cmpxchg(&uvm_ops, NULL, ops) != NULL)
		return -EEXIST;

	pr_info("gpu_mem_ops registered in nvidia-uvm\n");
	return 0;
}

/* struct_ops注销回调：当BPF程序被卸载时被调用 */
static void gpu_mem_ops_unreg(void *kdata, struct bpf_link *link)
{
	struct gpu_mem_ops *ops = kdata;

	if (cmpxchg(&uvm_ops, ops, NULL) != ops) {
		pr_warn("gpu_mem_ops: unexpected unreg in nvidia-uvm\n");
		return;
	}

	pr_info("gpu_mem_ops unregistered from nvidia-uvm\n");
}

/* 定义完整的struct_ops操作集 */
static struct bpf_struct_ops gpu_mem_ops_struct_ops = {
	.verifier_ops = &gpu_mem_ops_verifier_ops, // 验证器
	.init = gpu_mem_ops_init,                   // 初始化
	.init_member = gpu_mem_ops_init_member,     // 成员初始化
	.reg = gpu_mem_ops_reg,                     // 注册
	.unreg = gpu_mem_ops_unreg,                 // 注销
	.cfi_stubs = &__bpf_ops_gpu_mem_ops,        // CFI存根
	.name = "gpu_mem_ops",                      // 名称
	.owner = THIS_MODULE,                       // 所有者
};

/* --- Proc文件系统接口 --- */

/* proc文件写入处理函数，用于手动触发BPF程序执行 */
static ssize_t trigger_write(struct file *file, const char __user *buf,
			     size_t count, loff_t *pos)
{
	struct gpu_mem_ops *ops;
	char kbuf[64];
	int ret = 0;

	// 将用户态数据拷贝到内核态
	if (count >= sizeof(kbuf))
		count = sizeof(kbuf) - 1;
	if (copy_from_user(kbuf, buf, count))
		return -EFAULT;
	kbuf[count] = '\0';

	// 使用RCU机制安全地读取全局ops指针
	rcu_read_lock();
	ops = rcu_dereference(uvm_ops);
	if (ops) {
		pr_info("UVM: Calling struct_ops callbacks:\n");
		// 调用注册的test_trigger回调
		if (ops->gpu_test_trigger) {
			ret = ops->gpu_test_trigger(kbuf, count);
			pr_info("UVM: gpu_test_trigger() returned: %d\n", ret);
		}
	} else {
		pr_info("UVM: No struct_ops registered\n");
	}
	rcu_read_unlock();

	return count;
}

static const struct proc_ops trigger_proc_ops = {
	.proc_write = trigger_write, // 关联写入操作
};

/* --- 模块生命周期管理 --- */

int uvm_bpf_struct_ops_init(void)
{
	int ret;

	/* 1. 注册kfunc ID集合，这样BPF程序就能调用我们定义的内核函数 */
	ret = register_btf_kfunc_id_set(BPF_PROG_TYPE_STRUCT_OPS, &uvm_bpf_kfunc_set);
	if (ret) {
		pr_err("UVM: Failed to register BTF kfunc ID set: %d\n", ret);
		return ret;
	}
	pr_info("UVM: kfunc ID set registered successfully\n");

	// 尝试也为KPROBE程序注册，允许它们调用gpu kfuncs (非致命错误)
	ret = register_btf_kfunc_id_set(BPF_PROG_TYPE_KPROBE, &uvm_bpf_kfunc_set);
	if (ret) {
		pr_warn("UVM: Failed to register kfunc for kprobe progs: %d (non-fatal)\n", ret);
		/* Non-fatal: struct_ops still works */
	}

	/* 2. 注册struct_ops本身 */
	ret = register_bpf_struct_ops(&gpu_mem_ops_struct_ops, gpu_mem_ops);
	if (ret) {
		pr_err("UVM: Failed to register struct_ops: %d\n", ret);
		return ret;
	}

	/* 3. 创建proc文件，方便用户态触发测试 */
	trigger_file = proc_create("bpf_testmod_trigger", 0222, NULL, &trigger_proc_ops);
	if (!trigger_file)
		return -ENOMEM;

	pr_info("UVM: bpf_struct_ops initialized\n");
	return 0;
}

void uvm_bpf_struct_ops_exit(void)
{
	// 清理proc文件
	if (trigger_file)
		proc_remove(trigger_file);
	// 注意：struct_ops的注销由内核自动在模块卸载时完成
	pr_info("UVM: bpf_struct_ops cleaned up\n");
}

/* --- 内核调用BPF钩子的包装函数 --- */
// 这些函数提供了安全的API，供内核其他部分在需要时调用已注册的BPF程序。

/**
 * @brief 调用GPU页面预取的BPF钩子函数
 */
enum uvm_bpf_action uvm_bpf_call_gpu_page_prefetch(
	uvm_page_index_t page_index,
	uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	uvm_va_block_region_t *max_prefetch_region,
	uvm_va_block_region_t *result_region)
{
	struct gpu_mem_ops *ops;
	int ret = UVM_BPF_ACTION_DEFAULT;

	// 使用RCU安全地读取ops指针
	rcu_read_lock();
	ops = rcu_dereference(uvm_ops);
	if (ops && ops->gpu_page_prefetch) {
		// 调用BPF程序并获取返回的动作码
		ret = ops->gpu_page_prefetch(page_index, bitmap_tree,
						       max_prefetch_region,
						       result_region);
	}
	rcu_read_unlock();

	return (enum uvm_bpf_action)ret;
}

/**
 * @brief 调用GPU页面预取迭代的BPF钩子函数
 */
enum uvm_bpf_action uvm_bpf_call_gpu_page_prefetch_iter(
	uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
	uvm_va_block_region_t *max_prefetch_region,
	uvm_va_block_region_t *current_region,
	unsigned int counter,
	uvm_va_block_region_t *prefetch_region)
{
	struct gpu_mem_ops *ops;
	int ret = UVM_BPF_ACTION_DEFAULT;

	rcu_read_lock();
	ops = rcu_dereference(uvm_ops);
	if (ops && ops->gpu_page_prefetch_iter) {
		ret = ops->gpu_page_prefetch_iter(bitmap_tree,
						     max_prefetch_region, current_region,
						     counter, prefetch_region);
	}
	rcu_read_unlock();

	return (enum uvm_bpf_action)ret;
}

/* PMM eviction policy hook wrappers */

/**
 * @brief 调用GPU块激活的BPF钩子函数
 */
void uvm_bpf_call_gpu_block_activate(
	uvm_pmm_gpu_t *pmm,
	uvm_gpu_chunk_t *chunk,
	struct list_head *list)
{
	struct gpu_mem_ops *ops;

	rcu_read_lock();
	ops = rcu_dereference(uvm_ops);
	if (ops && ops->gpu_block_activate) {
		// 此类钩子不返回动作码，直接调用
		ops->gpu_block_activate(pmm, chunk, list);
	}
	rcu_read_unlock();
}

/**
 * @brief 调用GPU块访问的BPF钩子函数
 */
enum uvm_bpf_action uvm_bpf_call_gpu_block_access(
	uvm_pmm_gpu_t *pmm,
	uvm_gpu_chunk_t *chunk,
	struct list_head *list)
{
	struct gpu_mem_ops *ops;
	int ret = UVM_BPF_ACTION_DEFAULT;

	rcu_read_lock();
	ops = rcu_dereference(uvm_ops);
	if (ops && ops->gpu_block_access) {
		ret = ops->gpu_block_access(pmm, chunk, list);
	}
	rcu_read_unlock();

	return (enum uvm_bpf_action)ret;
}

/**
 * @brief 调用GPU驱逐准备的BPF钩子函数
 */
void uvm_bpf_call_gpu_evict_prepare(
	uvm_pmm_gpu_t *pmm,
	struct list_head *va_block_used,
	struct list_head *va_block_unused)
{
	struct gpu_mem_ops *ops;

	rcu_read_lock();
	ops = rcu_dereference(uvm_ops);
	if (ops && ops->gpu_evict_prepare) {
		ops->gpu_evict_prepare(pmm, va_block_used, va_block_unused);
	}
	rcu_read_unlock();
}