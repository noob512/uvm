/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Attention-Aware Eviction — Userspace Loader
 *
 * Loads the BPF struct_ops program and pins score_map / stats_map to
 * /sys/fs/bpf/ so the Python score_bridge daemon can populate scores
 * and read statistics.
 *
 * Usage:
 *   sudo ./attention_aware_eviction [--stats-interval N]
 *
 * Then run score_bridge.py in another terminal to populate attention scores.
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "attention_aware_eviction.skel.h"
#include "cleanup_struct_ops.h"

#define SCORE_MAP_PIN_PATH  "/sys/fs/bpf/attention_score_map"
#define STATS_MAP_PIN_PATH  "/sys/fs/bpf/attention_stats_map"

#define STAT_NUM_COUNTERS 8

static const char *stat_names[STAT_NUM_COUNTERS] = {
	[0] = "activate_total",
	[1] = "score_hit",
	[2] = "move_head_trash",
	[3] = "move_tail_hot",
	[4] = "tier_cool",
	[5] = "t1_protect",
	[6] = "score_miss",
	[7] = "(reserved)",
};

static int libbpf_print_fn(enum libbpf_print_level level,
			    const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void handle_signal(int sig)
{
	exiting = true;
}

static void unpin_maps(void)
{
	unlink(SCORE_MAP_PIN_PATH);
	unlink(STATS_MAP_PIN_PATH);
}

static int verify_maps_unpinned(const char *label)
{
	int err = 0;

	if (access(SCORE_MAP_PIN_PATH, F_OK) == 0) {
		fprintf(stderr, "[%s] score_map pin still exists: %s\n",
			label, SCORE_MAP_PIN_PATH);
		err = -EEXIST;
	}

	if (access(STATS_MAP_PIN_PATH, F_OK) == 0) {
		fprintf(stderr, "[%s] stats_map pin still exists: %s\n",
			label, STATS_MAP_PIN_PATH);
		err = -EEXIST;
	}

	if (!err) {
		printf("[%s] Verified: no pinned attention-aware maps remain.\n",
		       label);
	}

	return err;
}

static void print_stats(int stats_fd, int num_cpus)
{
	int i, c;

	for (i = 0; i < STAT_NUM_COUNTERS - 1; i++) {
		__u32 key = i;
		__u64 values[256];
		__u64 total = 0;

		memset(values, 0, sizeof(values));
		if (bpf_map_lookup_elem(stats_fd, &key, values) == 0) {
			for (c = 0; c < num_cpus && c < 256; c++)
				total += values[c];
			printf("  %-22s %llu\n", stat_names[i],
			       (unsigned long long)total);
		}
	}
}

static void print_usage(const char *prog)
{
	printf("Usage: %s [OPTIONS]\n\n", prog);
	printf("Attention-Aware Eviction Policy Loader\n\n");
	printf("Pins BPF maps for score_bridge.py:\n");
	printf("  score_map → %s\n", SCORE_MAP_PIN_PATH);
	printf("  stats_map → %s\n\n", STATS_MAP_PIN_PATH);
	printf("Options:\n");
	printf("  -s, --stats-interval N  Print stats every N seconds (0=off, default: 10)\n");
	printf("  -h, --help              Show this help\n");
}

int main(int argc, char **argv)
{
	struct attention_aware_eviction_bpf *skel;
	struct bpf_link *link = NULL;
	int err, stats_interval = 10;
	int verify_err;

	static struct option long_options[] = {
		{"stats-interval", required_argument, 0, 's'},
		{"help",           no_argument,       0, 'h'},
		{0, 0, 0, 0}
	};

	int c;
	while ((c = getopt_long(argc, argv, "s:h", long_options, NULL)) != -1) {
		switch (c) {
		case 's': stats_interval = atoi(optarg); break;
		case 'h': print_usage(argv[0]); return 0;
		default:  print_usage(argv[0]); return 1;
		}
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	err = cleanup_old_struct_ops();
	if (err == -EEXIST) {
		fprintf(stderr,
			"Refusing to load attention_aware_eviction: stale UVM struct_ops instances are still present.\n");
		return 1;
	}

	err = verify_no_uvm_struct_ops_instances("startup");
	if (err) {
		fprintf(stderr,
			"Refusing to load attention_aware_eviction: unable to verify a clean UVM struct_ops state.\n");
		return 1;
	}

	/* Remove stale pins from previous runs */
	unpin_maps();
	err = verify_maps_unpinned("startup");
	if (err)
		return 1;

	skel = attention_aware_eviction_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = attention_aware_eviction_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* Pin maps so score_bridge.py can access them */
	err = bpf_map__pin(skel->maps.score_map, SCORE_MAP_PIN_PATH);
	if (err) {
		fprintf(stderr, "Failed to pin score_map at %s: %s\n",
			SCORE_MAP_PIN_PATH, strerror(-err));
		goto cleanup;
	}

	err = bpf_map__pin(skel->maps.stats_map, STATS_MAP_PIN_PATH);
	if (err) {
		fprintf(stderr, "Failed to pin stats_map at %s: %s\n",
			STATS_MAP_PIN_PATH, strerror(-err));
		goto cleanup;
	}

	/* Attach struct_ops */
	link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_attention_aware);
	if (!link) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n",
			strerror(-err), err);
		goto cleanup;
	}

	printf("\n");
	printf("=== Attention-Aware Eviction Policy ===\n");
	printf("  Prefetch : always_max (full VA block)\n");
	printf("  Eviction : score-based (KV cache) + T1 frequency (fallback)\n");
	printf("  score_map: %s\n", SCORE_MAP_PIN_PATH);
	printf("  stats_map: %s\n", STATS_MAP_PIN_PATH);
	printf("\n");
	printf("Run score_bridge.py to populate attention scores.\n");
	printf("Press Ctrl-C to exit.\n\n");

	{
		int stats_fd = bpf_map__fd(skel->maps.stats_map);
		int num_cpus = libbpf_num_possible_cpus();
		int elapsed = 0;

		while (!exiting) {
			sleep(1);
			elapsed++;
			if (stats_interval > 0 &&
			    elapsed % stats_interval == 0) {
				printf("--- Stats (t=%ds) ---\n", elapsed);
				print_stats(stats_fd, num_cpus);
				printf("\n");
			}
		}
	}

	printf("Detaching struct_ops...\n");
	bpf_link__destroy(link);
	link = NULL;

cleanup:
	unpin_maps();
	verify_err = verify_maps_unpinned("shutdown");
	if (!err && verify_err)
		err = verify_err;
	verify_err = verify_no_uvm_struct_ops_instances("shutdown");
	if (!err && verify_err)
		err = verify_err;
	if (link)
		bpf_link__destroy(link);
	attention_aware_eviction_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
