/* SPDX-License-Identifier: GPL-2.0 */
/*
 * QoS-Driven Eviction Loader
 *
 * Loads prefetch_always_max_qos.bpf.c, pins shared maps for sched_ext,
 * populates LC PID array, and reports controller state.
 *
 * Usage:
 *   ./prefetch_always_max_qos [-l PID ...] [-t TARGET] [-v]
 *
 *   -l PID    Register an LC (latency-critical) process.
 *             LC pages are protected from eviction when under pressure.
 *   -t RATE   Target LC fault rate (default: 200 faults/sec).
 *             Lower = stricter LC protection, more BE eviction.
 *   -v        Verbose output (1-second stats instead of 5-second)
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

/* Kernel types used by BPF skeleton's rodata/bss structs */
typedef unsigned int u32;
typedef unsigned long long u64;

#include "prefetch_always_max_qos.skel.h"
#include "cleanup_struct_ops.h"
#include "shared_maps.h"

#define MAX_LC_PIDS 16

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
	exiting = true;
}

static void read_stats(struct prefetch_always_max_qos_bpf *skel, __u64 *out)
{
	int nr_cpus = libbpf_num_possible_cpus();
	if (nr_cpus <= 0)
		return;

	memset(out, 0, sizeof(out[0]) * 8);

	for (__u32 idx = 0; idx < 8; idx++) {
		__u64 cnts[nr_cpus];
		int ret;

		ret = bpf_map_lookup_elem(bpf_map__fd(skel->maps.stats),
					  &idx, cnts);
		if (ret < 0)
			continue;
		for (int cpu = 0; cpu < nr_cpus; cpu++)
			out[idx] += cnts[cpu];
	}
}

static void print_gpu_state(struct prefetch_always_max_qos_bpf *skel)
{
	int fd = bpf_map__fd(skel->maps.gpu_state_map);
	__u32 key = 0, next_key;
	struct gpu_pid_state state;

	while (bpf_map_get_next_key(fd, &key, &next_key) == 0) {
		if (bpf_map_lookup_elem(fd, &next_key, &state) == 0) {
			printf("  PID %-8u  fault_rate=%-5llu  evict=%-8llu  "
			       "used=%-8llu  thrashing=%s\n",
			       next_key,
			       (unsigned long long)state.fault_rate,
			       (unsigned long long)state.eviction_count,
			       (unsigned long long)state.used_count,
			       state.is_thrashing ? "YES" : "no");
		}
		key = next_key;
	}
}

int main(int argc, char **argv) {
	struct prefetch_always_max_qos_bpf *skel;
	struct bpf_link *link;
	int err, opt;
	__u32 lc_pids[MAX_LC_PIDS];
	int n_lc = 0;
	__u64 target_fr = 0; /* 0 = use default */
	bool verbose = false;

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);

	while ((opt = getopt(argc, argv, "l:t:vh")) != -1) {
		switch (opt) {
		case 'l':
			if (n_lc < MAX_LC_PIDS) {
				lc_pids[n_lc++] = (__u32)atoi(optarg);
			} else {
				fprintf(stderr, "Too many -l PIDs (max %d)\n",
					MAX_LC_PIDS);
			}
			break;
		case 't':
			target_fr = (__u64)atoll(optarg);
			break;
		case 'v':
			verbose = true;
			break;
		default:
			fprintf(stderr,
				"QoS-Driven Eviction: Feedback-controlled LC page protection\n\n"
				"Usage: %s [-l PID ...] [-t TARGET] [-v]\n\n"
				"  -l PID     LC (latency-critical) process PID.\n"
				"             LC pages protected from eviction when fault_rate > target.\n"
				"  -t RATE    Target LC fault rate (default: 200 faults/sec).\n"
				"             Lower = stricter protection, more BE eviction.\n"
				"  -v         Verbose (1s stats interval)\n"
				"  -h         Display this help\n",
				argv[0]);
			return opt != 'h';
		}
	}

	cleanup_old_struct_ops();

	skel = prefetch_always_max_qos_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	/* Set rodata before load */
	skel->rodata->n_lc_pids = (__u32)n_lc;
	if (target_fr > 0)
		skel->rodata->target_fault_rate = target_fr;

	err = prefetch_always_max_qos_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* Populate lc_pid_array */
	if (n_lc > 0) {
		int arr_fd = bpf_map__fd(skel->maps.lc_pid_array);
		for (int i = 0; i < n_lc; i++) {
			__u32 idx = (__u32)i;
			err = bpf_map_update_elem(arr_fd, &idx,
						  &lc_pids[i], BPF_ANY);
			if (err)
				fprintf(stderr,
					"WARNING: Failed to add LC PID %u: %s\n",
					lc_pids[i], strerror(-err));
			else
				printf("Registered LC PID %u\n", lc_pids[i]);
		}
	}

	/* Pin gpu_state_map for sched_ext to read */
	int gpu_state_fd = bpf_map__fd(skel->maps.gpu_state_map);
	unlink(XCOORD_GPU_STATE_PIN);
	err = bpf_obj_pin(gpu_state_fd, XCOORD_GPU_STATE_PIN);
	if (err) {
		fprintf(stderr, "Failed to pin gpu_state_map at %s: %s\n",
			XCOORD_GPU_STATE_PIN, strerror(errno));
	} else {
		printf("Pinned gpu_state_map at %s\n", XCOORD_GPU_STATE_PIN);
	}

	/* Pin uvm_worker_pids map for sched_ext to read */
	int workers_fd = bpf_map__fd(skel->maps.uvm_worker_pids);
	unlink(XCOORD_UVM_WORKERS_PIN);
	err = bpf_obj_pin(workers_fd, XCOORD_UVM_WORKERS_PIN);
	if (err) {
		fprintf(stderr, "Failed to pin uvm_worker_pids at %s: %s\n",
			XCOORD_UVM_WORKERS_PIN, strerror(errno));
	} else {
		printf("Pinned uvm_worker_pids at %s\n", XCOORD_UVM_WORKERS_PIN);
	}

	/* Attach struct_ops */
	link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_always_max_qos);
	if (!link) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n",
			strerror(-err), err);
		unlink(XCOORD_GPU_STATE_PIN);
		unlink(XCOORD_UVM_WORKERS_PIN);
		goto cleanup;
	}

	printf("\nQoS-Driven Eviction loaded!\n");
	printf("  Prefetch: always_max\n");
	printf("  Eviction: cycle_moe + QoS LC protection\n");
	printf("  Target LC fault rate: %llu faults/sec\n",
	       (unsigned long long)skel->rodata->target_fault_rate);
	printf("  Regulate interval: %llums\n",
	       (unsigned long long)skel->rodata->regulate_interval_ns / 1000000);
	printf("  LC PIDs: %d registered\n", n_lc);
	printf("  ki_gain=%llu, decay=>>%u, max_integral=%llu\n",
	       (unsigned long long)skel->rodata->ki_gain,
	       skel->rodata->decay_shift,
	       (unsigned long long)skel->rodata->max_integral);
	printf("  Shared maps: %s, %s\n",
	       XCOORD_GPU_STATE_PIN, XCOORD_UVM_WORKERS_PIN);
	printf("\nPress Ctrl-C to exit...\n\n");

	while (!exiting) {
		__u64 stats[8];

		read_stats(skel, stats);
		printf("activate=%llu used=%llu t1=%llu lc_prot=%llu "
		       "evict=%llu | bias=%u%% integral=%llu lc_fr=%llu "
		       "regulate=%llu be_demoted=%llu\n",
		       (unsigned long long)stats[0], /* ACTIVATE */
		       (unsigned long long)stats[1], /* USED */
		       (unsigned long long)stats[2], /* T1_PROTECTED */
		       (unsigned long long)stats[3], /* LC_PROTECTED */
		       (unsigned long long)stats[4], /* EVICT */
		       (unsigned int)(skel->bss->eviction_bias / 10),
		       (unsigned long long)skel->bss->eviction_integral,
		       (unsigned long long)skel->bss->lc_fault_rate_observed,
		       (unsigned long long)stats[5], /* REGULATE */
		       (unsigned long long)stats[6]  /* BE_DEMOTED */);

		if (verbose) {
			print_gpu_state(skel);
		}

		fflush(stdout);
		sleep(verbose ? 1 : 5);
	}

	printf("\nDetaching...\n");
	print_gpu_state(skel);
	bpf_link__destroy(link);

	/* Unpin shared maps on exit */
	unlink(XCOORD_GPU_STATE_PIN);
	unlink(XCOORD_UVM_WORKERS_PIN);
	printf("Unpinned shared maps\n");

cleanup:
	prefetch_always_max_qos_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
