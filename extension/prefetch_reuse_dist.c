/* N2: Reuse Distance Eviction - Loader
 *
 * Usage: sudo ./prefetch_reuse_dist [short_reuse_ms] [enable_xb]
 *   short_reuse_ms: threshold in ms (default 50)
 *   enable_xb: 0=no cross-block, 1=direction-aware XB (default 0)
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_reuse_dist.skel.h"
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
	struct prefetch_reuse_dist_bpf *skel;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	int err;
	int opt_reuse_ms = 50;
	int opt_xb = 0;

	if (argc >= 2)
		opt_reuse_ms = atoi(argv[1]);
	if (argc >= 3)
		opt_xb = atoi(argv[2]);

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_reuse_dist_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	skel->rodata->short_reuse_ns = (__u64)opt_reuse_ms * 1000000ULL;
	skel->rodata->enable_xb = opt_xb;

	err = prefetch_reuse_dist_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_reuse_dist);
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

	printf("Loaded: N2 Reuse Distance Eviction\n");
	printf("  Prefetch: always_max%s\n", opt_xb ? " + direction-aware XB" : "");
	printf("  Eviction: reuse distance (threshold=%dms)\n", opt_reuse_ms);
	printf("Press Ctrl-C to exit...\n");

	while (!exiting)
		sleep(2);

	printf("\n=== Reuse Distance Stats ===\n");
	int stats_fd = bpf_map__fd(skel->maps.rd_stats);
	if (stats_fd >= 0) {
		const char *stat_names[] = {
			"kprobe fires", "prefetch_hook fires",
			"wq scheduled", "wq callback ran",
			"migrate success", "migrate failed",
			"rate-limit skip", "wq callback skip",
			"direction skip", "dedup skip",
			"RD protect (short reuse)", "RD expose (long reuse)",
		};
		for (__u32 i = 0; i < 12; i++) {
			__u64 val = 0;
			bpf_map_lookup_elem(stats_fd, &i, &val);
			printf("  %-30s %llu\n", stat_names[i], (unsigned long long)val);
		}
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_kprobe) bpf_link__destroy(link_kprobe);
	if (link_struct_ops) bpf_link__destroy(link_struct_ops);
	prefetch_reuse_dist_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
