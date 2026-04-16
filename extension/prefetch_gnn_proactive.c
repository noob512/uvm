#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/types.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_gnn_proactive.skel.h"
#include "cleanup_struct_ops.h"

#define DEFAULT_UVM_ALLOCATOR \
	"/home/yunwei37/workspace/gpu/gpu_ext/workloads/pytorch/uvm_allocator.so"
#define DEFAULT_CUDART \
	"/home/yunwei37/workspace/gpu/gpu_ext/workloads/pytorch/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
#define MAX_PREFETCH_BLOCKS 8

struct feature_tensor_info {
	__u64 addr;
	__u64 size;
	__u32 tgid;
	__u32 reserved;
};

struct sync_state {
	__u64 sync_calls;
	__u64 direct_prefetches;
};

static const char *const stat_names[] = {
	"kprobe fires",
	"large alloc entry",
	"feature tensor tracked",
	"cuda sync hits",
	"direct proactive migrate ok",
	"direct proactive migrate fail",
	"sync without feature",
	"sync without va_space",
	"pending request set",
	"pending request drained",
	"pending wq scheduled",
	"wq callback ran",
	"wq migrate ok",
	"wq migrate fail",
	"prefetch hook fires",
	"xb wq scheduled",
	"xb direction skip",
	"xb dedup skip",
	"xb rate-limit skip",
	"cycle_moe T1 protects",
};

static volatile bool exiting;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	if (level == LIBBPF_DEBUG)
		return 0;
	return vfprintf(stderr, format, args);
}

static void handle_signal(int sig)
{
	exiting = true;
}

static void usage(const char *prog)
{
	fprintf(stderr, "Usage: %s <path-to-uvm_allocator.so> [prefetch_blocks]\n", prog);
	fprintf(stderr, "Default allocator: %s\n", DEFAULT_UVM_ALLOCATOR);
	fprintf(stderr, "Default prefetch_blocks: %d (max %d)\n",
		MAX_PREFETCH_BLOCKS, MAX_PREFETCH_BLOCKS);
}

static int read_u64_map(int fd, __u32 key, __u64 *value)
{
	return bpf_map_lookup_elem(fd, &key, value);
}

static int attach_uprobe_symbol(struct bpf_program *prog, const char *path,
				const char *symbol, bool retprobe,
				struct bpf_link **link_out)
{
	LIBBPF_OPTS(bpf_uprobe_opts, opts,
		.func_name = symbol,
		.retprobe = retprobe,
	);
	struct bpf_link *link;
	int err;

	link = bpf_program__attach_uprobe_opts(prog, -1, path, 0, &opts);
	err = libbpf_get_error(link);
	if (err) {
		errno = -err;
		return err;
	}

	*link_out = link;
	return 0;
}

int main(int argc, char **argv)
{
	struct prefetch_gnn_proactive_bpf *skel = NULL;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	struct bpf_link *link_malloc_enter = NULL;
	struct bpf_link *link_malloc_ret = NULL;
	struct bpf_link *link_cuda_sync = NULL;
	const char *allocator_path = DEFAULT_UVM_ALLOCATOR;
	const char *cudart_path = DEFAULT_CUDART;
	__u32 blocks = MAX_PREFETCH_BLOCKS;
	int err = 0;

	if (argc >= 2) {
		if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
			usage(argv[0]);
			return 0;
		}
		allocator_path = argv[1];
	}
	if (argc >= 3) {
		long parsed = strtol(argv[2], NULL, 10);

		if (parsed <= 0) {
			fprintf(stderr, "Invalid prefetch_blocks: %s\n", argv[2]);
			return 1;
		}
		if (parsed > MAX_PREFETCH_BLOCKS)
			parsed = MAX_PREFETCH_BLOCKS;
		blocks = (__u32)parsed;
	}
	if (argc >= 4)
		cudart_path = argv[3];

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_gnn_proactive_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	skel->rodata->prefetch_blocks = blocks;

	err = prefetch_gnn_proactive_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_gnn_proactive);
	err = libbpf_get_error(link_struct_ops);
	if (err) {
		link_struct_ops = NULL;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("struct_ops attached (always_max + XB + cycle_moe)\n");

	link_kprobe = bpf_program__attach(skel->progs.capture_va_block);
	err = libbpf_get_error(link_kprobe);
	if (err) {
		link_kprobe = NULL;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("kprobe attached (va_space capture)\n");

	err = attach_uprobe_symbol(skel->progs.uvm_malloc_enter, allocator_path,
				   "uvm_malloc", false, &link_malloc_enter);
	if (err) {
		fprintf(stderr, "Failed to attach uprobe on %s:uvm_malloc: %s (%d)\n",
			allocator_path, strerror(errno), err);
		goto cleanup;
	}
	printf("uprobe attached: %s:uvm_malloc\n", allocator_path);

	err = attach_uprobe_symbol(skel->progs.uvm_malloc_ret, allocator_path,
				   "uvm_malloc", true, &link_malloc_ret);
	if (err) {
		fprintf(stderr, "Failed to attach uretprobe on %s:uvm_malloc: %s (%d)\n",
			allocator_path, strerror(errno), err);
		goto cleanup;
	}
	printf("uretprobe attached: %s:uvm_malloc\n", allocator_path);

	err = attach_uprobe_symbol(skel->progs.cuda_sync_epoch_boundary, cudart_path,
				   "cudaDeviceSynchronize", false, &link_cuda_sync);
	if (err) {
		fprintf(stderr, "Failed to attach uprobe on %s:cudaDeviceSynchronize: %s (%d)\n",
			cudart_path, strerror(errno), err);
		goto cleanup;
	}
	printf("uprobe attached: %s:cudaDeviceSynchronize\n", cudart_path);

	printf("\n=== Transparent GNN Proactive Prefetch ===\n");
	printf("  Allocator library: %s\n", allocator_path);
	printf("  CUDA runtime: %s\n", cudart_path);
	printf("  Feature tensor tracking: uvm_malloc(size > 4GB)\n");
	printf("  Epoch marker: cudaDeviceSynchronize()\n");
	printf("  Proactive prefix: first %u blocks (%u MB)\n",
	       blocks, blocks * 2);
	printf("  Intra-block: always_max\n");
	printf("  Cross-block: direction-aware adjacent block\n");
	printf("  Eviction: cycle_moe\n");
	printf("Press Ctrl-C to exit...\n\n");

	int feature_fd = bpf_map__fd(skel->maps.feature_map);
	int sync_fd = bpf_map__fd(skel->maps.sync_state_map);
	int stats_fd = bpf_map__fd(skel->maps.gnn_stats);

	while (!exiting) {
		__u32 key = 0;
		struct feature_tensor_info feature = {};
		struct sync_state sync_state = {};
		__u64 stat_sync = 0;
		__u64 stat_direct = 0;
		__u64 stat_pending = 0;
		__u64 stat_pending_drain = 0;
		__u64 stat_xb = 0;
		__u64 stat_wq_ok = 0;

		sleep(2);

		if (feature_fd >= 0)
			bpf_map_lookup_elem(feature_fd, &key, &feature);
		if (sync_fd >= 0)
			bpf_map_lookup_elem(sync_fd, &key, &sync_state);
		if (stats_fd >= 0) {
			key = 3; read_u64_map(stats_fd, key, &stat_sync);
			key = 4; read_u64_map(stats_fd, key, &stat_direct);
			key = 8; read_u64_map(stats_fd, key, &stat_pending);
			key = 9; read_u64_map(stats_fd, key, &stat_pending_drain);
			key = 15; read_u64_map(stats_fd, key, &stat_xb);
			key = 12; read_u64_map(stats_fd, key, &stat_wq_ok);
		}

		printf("Feature: ");
		if (feature.addr) {
			printf("tgid=%u addr=0x%llx size=%.2f GB | ",
			       feature.tgid,
			       (unsigned long long)feature.addr,
			       (double)feature.size / 1e9);
		} else {
			printf("not captured yet | ");
		}

		printf("sync=%llu direct=%llu pending=%llu drained=%llu xb=%llu wq_ok=%llu total_direct=%llu\n",
		       (unsigned long long)stat_sync,
		       (unsigned long long)stat_direct,
		       (unsigned long long)stat_pending,
		       (unsigned long long)stat_pending_drain,
		       (unsigned long long)stat_xb,
		       (unsigned long long)stat_wq_ok,
		       (unsigned long long)sync_state.direct_prefetches);
	}

	printf("\n=== Final GNN Proactive Stats ===\n");
	if (stats_fd >= 0) {
		for (__u32 i = 0; i < sizeof(stat_names) / sizeof(stat_names[0]); i++) {
			__u64 value = 0;

			read_u64_map(stats_fd, i, &value);
			printf("  %-34s %llu\n", stat_names[i],
			       (unsigned long long)value);
		}
	}

	if (feature_fd >= 0) {
		__u32 key = 0;
		struct feature_tensor_info feature = {};

		if (bpf_map_lookup_elem(feature_fd, &key, &feature) == 0 && feature.addr) {
			printf("\nTracked feature tensor: tgid=%u addr=0x%llx size=%.2f GB\n",
			       feature.tgid,
			       (unsigned long long)feature.addr,
			       (double)feature.size / 1e9);
		}
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_cuda_sync)
		bpf_link__destroy(link_cuda_sync);
	if (link_malloc_ret)
		bpf_link__destroy(link_malloc_ret);
	if (link_malloc_enter)
		bpf_link__destroy(link_malloc_enter);
	if (link_kprobe)
		bpf_link__destroy(link_kprobe);
	if (link_struct_ops)
		bpf_link__destroy(link_struct_ops);
	prefetch_gnn_proactive_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
