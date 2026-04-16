/* Fault-Rate Throttled Cross-Block Prefetch - Loader
 *
 * Usage: sudo ./prefetch_throttled_xb [window_ms] [fault_threshold]
 *   window_ms: fault counting window in ms (default 1)
 *   fault_threshold: skip XB if > N faults/window (default 50)
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_throttled_xb.skel.h"
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
	struct prefetch_throttled_xb_bpf *skel;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	int err;
	int opt_window_ms = 1;
	int opt_threshold = 50;

	if (argc >= 2)
		opt_window_ms = atoi(argv[1]);
	if (argc >= 3)
		opt_threshold = atoi(argv[2]);

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_throttled_xb_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	skel->rodata->window_ns = (__u64)opt_window_ms * 1000000ULL;
	skel->rodata->fault_threshold = opt_threshold;

	err = prefetch_throttled_xb_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_throttled_xb);
	if (!link_struct_ops) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}

	link_kprobe = bpf_program__attach(skel->progs.capture_va_block);
	if (!link_kprobe) {
		err = -errno;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}

	printf("Loaded: Fault-Rate Throttled Cross-Block\n");
	printf("  Intra-block: always_max\n");
	printf("  Cross-block: direction-aware, throttled (window=%dms, threshold=%d)\n",
	       opt_window_ms, opt_threshold);
	printf("  Eviction: cycle_moe\n");
	printf("Press Ctrl-C to exit...\n");

	while (!exiting)
		sleep(2);

	printf("\n=== Throttled XB Stats ===\n");
	int stats_fd = bpf_map__fd(skel->maps.txb_stats);
	if (stats_fd >= 0) {
		const char *stat_names[] = {
			"kprobe fires", "prefetch_hook fires",
			"wq scheduled", "wq callback ran",
			"migrate success", "migrate failed",
			"rate-limit skip", "wq callback skip",
			"direction skip", "dedup skip",
			"throttle skip (high fault rate)", "XB allowed (low fault rate)",
		};
		for (__u32 i = 0; i < 12; i++) {
			__u64 val = 0;
			bpf_map_lookup_elem(stats_fd, &i, &val);
			printf("  %-36s %llu\n", stat_names[i], (unsigned long long)val);
		}
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_kprobe) bpf_link__destroy(link_kprobe);
	if (link_struct_ops) bpf_link__destroy(link_struct_ops);
	prefetch_throttled_xb_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
