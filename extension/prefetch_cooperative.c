/* N3: Cooperative Prefetch-Eviction - Loader
 *
 * Usage: sudo ./prefetch_cooperative [protect_radius]
 *   protect_radius: number of VA blocks around XB target to protect (default 2)
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_cooperative.skel.h"
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
	struct prefetch_cooperative_bpf *skel;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	int err;
	int opt_radius = 2;

	if (argc >= 2)
		opt_radius = atoi(argv[1]);

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_cooperative_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	skel->rodata->protect_radius = opt_radius;

	err = prefetch_cooperative_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_cooperative);
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

	printf("Loaded: N3 Cooperative Prefetch-Eviction\n");
	printf("  Intra-block: always_max\n");
	printf("  Cross-block: direction-aware (1 block)\n");
	printf("  Eviction: cycle_moe + cooperative protection (radius=%d blocks)\n", opt_radius);
	printf("Press Ctrl-C to exit...\n");

	while (!exiting)
		sleep(2);

	printf("\n=== Cooperative Stats ===\n");
	int stats_fd = bpf_map__fd(skel->maps.coop_stats);
	if (stats_fd >= 0) {
		const char *stat_names[] = {
			"kprobe fires", "prefetch_hook fires",
			"wq scheduled", "wq callback ran",
			"migrate success", "migrate failed",
			"rate-limit skip", "wq callback skip",
			"direction skip", "dedup skip",
			"cooperative protect", "T1 protect",
		};
		for (__u32 i = 0; i < 12; i++) {
			__u64 val = 0;
			bpf_map_lookup_elem(stats_fd, &i, &val);
			printf("  %-24s %llu\n", stat_names[i], (unsigned long long)val);
		}
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_kprobe) bpf_link__destroy(link_kprobe);
	if (link_struct_ops) bpf_link__destroy(link_struct_ops);
	prefetch_cooperative_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
