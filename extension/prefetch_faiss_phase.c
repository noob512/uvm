/* FAISS Phase-Adaptive Cross-Block Prefetch - Loader
 *
 * Auto-detects BUILD (sequential K-means) vs SEARCH (random query) phases.
 * Cross-block prefetch is only enabled during BUILD phase.
 *
 * Hardcoded: cycle_moe eviction, 2MB cross-block prefetch, always_max intra-block.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_faiss_phase.skel.h"
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
	struct prefetch_faiss_phase_bpf *skel;
	struct bpf_link *link;
	struct bpf_link *kprobe_link;
	int err;

	(void)argc;
	(void)argv;

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_faiss_phase_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = prefetch_faiss_phase_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* Attach kprobe for va_block context capture */
	kprobe_link = bpf_program__attach(skel->progs.capture_va_block);
	if (!kprobe_link) {
		err = -errno;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}

	link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_faiss_phase);
	if (!link) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
		bpf_link__destroy(kprobe_link);
		goto cleanup;
	}

	printf("Loaded: FAISS phase-adaptive cross-block prefetch\n");
	printf("  Intra-block: always_max (prefetch entire VA block)\n");
	printf("  Cross-block: 2MB, BUILD phase only (sequential stride)\n");
	printf("  Phase detection: sliding window of 32, BUILD>=%d%%, SEARCH<=%d%%\n",
	       (16 * 100) / 32, (8 * 100) / 32);
	printf("  Eviction: default LRU (kernel handles all)\n");
	printf("Press Ctrl-C to exit...\n");

	while (!exiting) {
		sleep(1);
	}

	/* Print debug stats before detaching */
	printf("\n=== FAISS Phase-Adaptive Stats ===\n");
	int stats_fd = bpf_map__fd(skel->maps.fp_stats);
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
			"build_prefetch (XB during BUILD)",
			"search_skip (XB skipped, SEARCH)",
			"phase -> BUILD",
			"phase -> SEARCH",
		};
		for (__u32 i = 0; i < 12; i++) {
			__u64 val = 0;
			bpf_map_lookup_elem(stats_fd, &i, &val);
			printf("  %s: %llu\n", stat_names[i], val);
		}
	}

	/* Print final phase state */
	int phase_fd = bpf_map__fd(skel->maps.phase_map);
	if (phase_fd >= 0) {
		struct {
			__u8  stride_window[32];
			__u32 window_idx;
			__u32 seq_count;
			__u32 phase;
			__u64 last_block_end;
			__u64 phase_transitions;
		} ps = {};
		__u32 key = 0;
		if (bpf_map_lookup_elem(phase_fd, &key, &ps) == 0) {
			printf("\n  Final phase: %s\n",
			       ps.phase == 1 ? "BUILD" : "SEARCH");
			printf("  Sequential count: %u/%d\n", ps.seq_count, 32);
			printf("  Phase transitions: %llu\n", ps.phase_transitions);
			printf("  Last block end: 0x%llx\n", ps.last_block_end);
		}
	}

	printf("\nDetaching struct_ops...\n");
	bpf_link__destroy(link);
	bpf_link__destroy(kprobe_link);

cleanup:
	prefetch_faiss_phase_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
