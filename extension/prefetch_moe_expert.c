#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <linux/types.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_moe_expert.skel.h"
#include "cleanup_struct_ops.h"

#define DEFAULT_MIN_ALLOC_GB 10.0
#define DEFAULT_MIN_SYNC_GAP_NS 5000000ULL
#define VA_BLOCK_SIZE_MB 2ULL
#define BITMAP_WORDS 512ULL
#define BITMAP_BITS (BITMAP_WORDS * 64ULL)
#define BITMAP_COVERAGE_GB ((BITMAP_BITS * VA_BLOCK_SIZE_MB) / 1024.0)
#define ONE_GIB (1024.0 * 1024.0 * 1024.0)

struct model_buffer_info {
	__u64 base_addr;
	__u64 size;
	__u32 tgid;
	__u32 reserved;
};

struct token_stats {
	__u64 sync_count;
	__u64 prefetch_count;
	__u64 faults_recorded;
	__u64 prefetch_blocks;
};

enum debug_stat {
	DBG_RECORD_CALLED = 0,
	DBG_NO_MODEL,
	DBG_NO_MODEL_ADDR,
	DBG_OVERFLOW,
	DBG_NO_VA_BLOCK,
	DBG_NO_BLOCK_START,
	DBG_OUT_OF_RANGE,
	DBG_BITMAP_SET,
	DBG_BITMAP_DUP,
	DBG_MAX_STATS,
};

static const char *const debug_stat_names[DBG_MAX_STATS] = {
	[DBG_RECORD_CALLED] = "record_called",
	[DBG_NO_MODEL] = "no_model",
	[DBG_NO_MODEL_ADDR] = "no_model_addr",
	[DBG_OVERFLOW] = "overflow",
	[DBG_NO_VA_BLOCK] = "no_va_block",
	[DBG_NO_BLOCK_START] = "no_block_start",
	[DBG_OUT_OF_RANGE] = "out_of_range",
	[DBG_BITMAP_SET] = "bitmap_set",
	[DBG_BITMAP_DUP] = "bitmap_dup",
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
	fprintf(stderr, "Usage: %s <cudart.so> [min_alloc_size_gb]\n", prog);
	fprintf(stderr, "Default min_alloc_size_gb: %.0f\n", DEFAULT_MIN_ALLOC_GB);
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

static int parse_min_alloc_bytes(const char *arg, __u64 *bytes_out)
{
	char *end = NULL;
	double gb;
	long double bytes;

	errno = 0;
	gb = strtod(arg, &end);
	if (errno || !end || *end != '\0' || gb <= 0.0)
		return -EINVAL;

	bytes = (long double)gb * (long double)ONE_GIB;
	if (bytes > (long double)UINT64_MAX)
		return -ERANGE;

	*bytes_out = (__u64)bytes;
	return 0;
}

static void print_stats_line(const struct model_buffer_info *model,
			     const struct token_stats *stats,
			     __u32 target_tgid,
			     __u64 va_space)
{
	printf("target_tgid=%u model_base=0x%llx size=%.2f GB sync=%llu prefetch=%llu faults=%llu prefetch_blocks=%llu va_space=0x%llx\n",
	       target_tgid,
	       (unsigned long long)model->base_addr,
	       model->size ? (double)model->size / 1e9 : 0.0,
	       (unsigned long long)stats->sync_count,
	       (unsigned long long)stats->prefetch_count,
	       (unsigned long long)stats->faults_recorded,
	       (unsigned long long)stats->prefetch_blocks,
	       (unsigned long long)va_space);
}

static void print_debug_stats_line(int debug_fd, int first_block_fd)
{
	__u64 values[DBG_MAX_STATS] = {};
	__u64 block_start = 0;
	int i;

	if (debug_fd >= 0) {
		for (i = 0; i < DBG_MAX_STATS; i++) {
			__u32 key = i;

			bpf_map_lookup_elem(debug_fd, &key, &values[i]);
		}
	}

	if (first_block_fd >= 0) {
		__u32 key = 0;

		bpf_map_lookup_elem(first_block_fd, &key, &block_start);
	}

	printf("debug");
	for (i = 0; i < DBG_MAX_STATS; i++) {
		printf(" %s=%llu",
		       debug_stat_names[i],
		       (unsigned long long)values[i]);
	}
	printf(" first_block_start=0x%llx\n",
	       (unsigned long long)block_start);
}

int main(int argc, char **argv)
{
	struct prefetch_moe_expert_bpf *skel = NULL;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	struct bpf_link *link_malloc_enter = NULL;
	struct bpf_link *link_malloc_ret = NULL;
	struct bpf_link *link_cuda_sync = NULL;
	const char *cudart_path;
	__u64 min_alloc_bytes = (__u64)(DEFAULT_MIN_ALLOC_GB * ONE_GIB);
	__u64 min_sync_gap_ns = DEFAULT_MIN_SYNC_GAP_NS;
	int err = 0;

	if (argc < 2 || argc > 3) {
		if (argc >= 2 &&
		    (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
			usage(argv[0]);
			return 0;
		}

		usage(argv[0]);
		return 1;
	}

	cudart_path = argv[1];
	if (argc == 3) {
		err = parse_min_alloc_bytes(argv[2], &min_alloc_bytes);
		if (err) {
			fprintf(stderr, "Invalid min_alloc_size_gb: %s\n", argv[2]);
			return 1;
		}
	}

	if (access(cudart_path, R_OK) != 0) {
		fprintf(stderr, "Cannot read CUDA runtime: %s\n", cudart_path);
		return 1;
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_moe_expert_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	skel->rodata->min_alloc_size = min_alloc_bytes;
	skel->rodata->min_sync_gap_ns = min_sync_gap_ns;

	err = prefetch_moe_expert_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_moe_expert);
	err = libbpf_get_error(link_struct_ops);
	if (err) {
		link_struct_ops = NULL;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n",
			strerror(-err), err);
		goto cleanup;
	}

	link_kprobe = bpf_program__attach(skel->progs.capture_va_space);
	err = libbpf_get_error(link_kprobe);
	if (err) {
		link_kprobe = NULL;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n",
			strerror(-err), err);
		goto cleanup;
	}

	err = attach_uprobe_symbol(skel->progs.cuda_malloc_managed_enter,
				   cudart_path,
				   "cudaMallocManaged",
				   false,
				   &link_malloc_enter);
	if (err) {
		fprintf(stderr, "Failed to attach uprobe on %s:cudaMallocManaged: %s (%d)\n",
			cudart_path, strerror(errno), err);
		goto cleanup;
	}

	err = attach_uprobe_symbol(skel->progs.cuda_malloc_managed_ret,
				   cudart_path,
				   "cudaMallocManaged",
				   true,
				   &link_malloc_ret);
	if (err) {
		fprintf(stderr, "Failed to attach uretprobe on %s:cudaMallocManaged: %s (%d)\n",
			cudart_path, strerror(errno), err);
		goto cleanup;
	}

	err = attach_uprobe_symbol(skel->progs.cuda_sync_token_boundary,
				   cudart_path,
				   "cudaStreamSynchronize",
				   false,
				   &link_cuda_sync);
	if (err) {
		fprintf(stderr, "Failed to attach uprobe on %s:cudaStreamSynchronize: %s (%d)\n",
			cudart_path, strerror(errno), err);
		goto cleanup;
	}

	printf("struct_ops attached (always_max + cycle_moe + bitmap replay)\n");
	printf("kprobe attached (va_space capture)\n");
	printf("uprobe attached: %s:cudaMallocManaged\n", cudart_path);
	printf("uretprobe attached: %s:cudaMallocManaged\n", cudart_path);
	printf("uprobe attached: %s:cudaStreamSynchronize\n", cudart_path);
	printf("\n=== Transparent MoE Bitmap Prefetch ===\n");
	printf("  CUDA runtime: %s\n", cudart_path);
	printf("  Model tracking: largest cudaMallocManaged() >= %.2f GB\n",
	       (double)min_alloc_bytes / ONE_GIB);
	printf("  Bitmap coverage: %.0f GB (%llu x %llu MB blocks)\n",
	       BITMAP_COVERAGE_GB,
	       (unsigned long long)BITMAP_BITS,
	       (unsigned long long)VA_BLOCK_SIZE_MB);
	printf("  Token boundary: cudaStreamSynchronize() (min gap %.2f ms)\n",
	       (double)min_sync_gap_ns / 1e6);
	printf("  Intra-block prefetch: always_max\n");
	printf("  Eviction: cycle_moe (threshold=%d)\n", 3);
	printf("Press Ctrl-C to exit...\n\n");

	{
		int model_fd = bpf_map__fd(skel->maps.model_buffer_map);
		int stats_fd = bpf_map__fd(skel->maps.token_stats_map);
		int debug_fd = bpf_map__fd(skel->maps.debug_stats);
		int first_block_fd = bpf_map__fd(skel->maps.first_block_start);
		int target_fd = bpf_map__fd(skel->maps.target_tgid_map);
		int va_space_fd = bpf_map__fd(skel->maps.va_space_map);

		while (!exiting) {
			__u32 key = 0;
			struct model_buffer_info model = {};
			struct token_stats stats = {};
			__u32 target_tgid = 0;
			__u64 va_space = 0;

			sleep(2);

			if (model_fd >= 0)
				bpf_map_lookup_elem(model_fd, &key, &model);
			if (stats_fd >= 0)
				bpf_map_lookup_elem(stats_fd, &key, &stats);
			if (target_fd >= 0)
				bpf_map_lookup_elem(target_fd, &key, &target_tgid);
			if (va_space_fd >= 0)
				bpf_map_lookup_elem(va_space_fd, &key, &va_space);

			print_stats_line(&model, &stats, target_tgid, va_space);
			print_debug_stats_line(debug_fd, first_block_fd);
		}

		printf("\n=== Final MoE Bitmap Stats ===\n");
		{
			__u32 key = 0;
			struct model_buffer_info model = {};
			struct token_stats stats = {};
			__u32 target_tgid = 0;
			__u64 va_space = 0;

			if (model_fd >= 0)
				bpf_map_lookup_elem(model_fd, &key, &model);
			if (stats_fd >= 0)
				bpf_map_lookup_elem(stats_fd, &key, &stats);
			if (target_fd >= 0)
				bpf_map_lookup_elem(target_fd, &key, &target_tgid);
			if (va_space_fd >= 0)
				bpf_map_lookup_elem(va_space_fd, &key, &va_space);

			print_stats_line(&model, &stats, target_tgid, va_space);
			print_debug_stats_line(debug_fd, first_block_fd);
		}
	}

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
	prefetch_moe_expert_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
