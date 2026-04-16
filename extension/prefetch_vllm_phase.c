/* vLLM Uprobe Phase Detection - Loader
 *
 * Attaches uprobe on uvm_set_phase() in vLLM's uvm_allocator.so
 * to detect prefill vs decode phases.
 *
 * Usage:
 *   sudo ./prefetch_vllm_phase [uvm_allocator_path]
 *
 * Default path: workloads/vllm/vllm/vllm/uvm_allocator.abi3.so
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_vllm_phase.skel.h"
#include "cleanup_struct_ops.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
	exiting = true;
}

int main(int argc, char **argv) {
	struct prefetch_vllm_phase_bpf *skel;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	struct bpf_link *link_uprobe = NULL;
	int err;

	const char *allocator_path;
	int opt_decode_mode = 0;
	int opt_decode_radius = 32;
	int opt_xb_decode = 0;

	if (argc >= 2) {
		allocator_path = argv[1];
	} else {
		allocator_path = "/home/yunwei37/workspace/gpu/gpu_ext/workloads/vllm/vllm/vllm/uvm_allocator.abi3.so";
	}
	if (argc >= 3)
		opt_decode_mode = atoi(argv[2]);
	if (argc >= 4)
		opt_decode_radius = atoi(argv[3]);
	if (argc >= 5)
		opt_xb_decode = atoi(argv[4]);

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_vllm_phase_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	/* Set rodata before load */
	skel->rodata->decode_prefetch_mode = opt_decode_mode;
	skel->rodata->decode_radius = opt_decode_radius;
	skel->rodata->xb_decode_enable = opt_xb_decode;

	err = prefetch_vllm_phase_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* 1. Attach struct_ops */
	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_vllm_phase);
	if (!link_struct_ops) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("struct_ops attached (vllm_phase)\n");

	/* 2. Attach kprobe */
	link_kprobe = bpf_program__attach(skel->progs.capture_va_block);
	if (!link_kprobe) {
		err = -errno;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("kprobe attached (va_block capture)\n");

	/* 3. Attach uprobe on uvm_set_phase */
	LIBBPF_OPTS(bpf_uprobe_opts, uprobe_opts,
		.func_name = "uvm_set_phase",
		.retprobe = false,
	);
	link_uprobe = bpf_program__attach_uprobe_opts(
		skel->progs.vllm_set_phase,
		-1,
		allocator_path,
		0,
		&uprobe_opts
	);
	if (!link_uprobe) {
		err = -errno;
		fprintf(stderr, "Failed to attach uprobe on %s:uvm_set_phase: %s (%d)\n",
		        allocator_path, strerror(-err), err);
		goto cleanup;
	}
	printf("uprobe attached: uvm_set_phase -> phase detection\n");

	const char *mode_names[] = {"always_max", "narrow_region", "default_kernel", "forward_only"};
	printf("\n=== vLLM Uprobe Phase Detection ===\n");
	printf("  Library: %s\n", allocator_path);
	printf("  Prefill intra-block: always_max\n");
	printf("  Decode intra-block: %s (mode=%d, radius=%d)\n",
	       opt_decode_mode < 4 ? mode_names[opt_decode_mode] : "unknown",
	       opt_decode_mode, opt_decode_radius);
	printf("  Cross-block: direction-aware, %s\n",
	       opt_xb_decode ? "BOTH phases" : "PREFILL only");
	printf("  Eviction: cycle_moe (T1 protect + DEFAULT non-T1)\n");
	printf("Press Ctrl-C to exit...\n\n");

	int stats_fd = bpf_map__fd(skel->maps.vp_stats);
	int phase_fd = bpf_map__fd(skel->maps.phase_map);

	while (!exiting) {
		sleep(2);

		__u32 phase_key = 0, phase_val = 0;
		if (phase_fd >= 0)
			bpf_map_lookup_elem(phase_fd, &phase_key, &phase_val);

		printf("Phase: %s | ",
		       phase_val == 1 ? "PREFILL" :
		       phase_val == 2 ? "DECODE" : "UNKNOWN");

		if (stats_fd >= 0) {
			__u64 val;
			__u32 k;
			k = 10; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("prefill=%llu ", (unsigned long long)val);
			k = 11; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("decode=%llu ", (unsigned long long)val);
			k = 12; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("xb_prefill=%llu ", (unsigned long long)val);
			k = 13; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("decode_skip=%llu", (unsigned long long)val);
		}
		printf("\n");
	}

	printf("\n=== Final vLLM Phase Stats ===\n");
	if (stats_fd >= 0) {
		const char *stat_names[] = {
			"kprobe fires",
			"prefetch_hook fires",
			"wq scheduled",
			"wq callback ran",
			"migrate success",
			"migrate failed",
			"rate-limit skip",
			"wq callback skip",
			"direction skip",
			"dedup skip",
			"phase -> PREFILL",
			"phase -> DECODE",
			"prefill XB (cross-block during prefill)",
			"decode skip (XB skipped during decode)",
		};
		for (__u32 i = 0; i < 14; i++) {
			__u64 val = 0;
			bpf_map_lookup_elem(stats_fd, &i, &val);
			printf("  %-42s %llu\n", stat_names[i], (unsigned long long)val);
		}
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_uprobe) bpf_link__destroy(link_uprobe);
	if (link_kprobe) bpf_link__destroy(link_kprobe);
	if (link_struct_ops) bpf_link__destroy(link_struct_ops);
	prefetch_vllm_phase_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
