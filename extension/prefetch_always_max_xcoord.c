/* SPDX-License-Identifier: GPL-2.0 */
/*
 * xCoord GPU-side Loader: Always-Max Prefetch + Cycle-MoE + Shared State
 *
 * Loads prefetch_always_max_xcoord.bpf.c and pins shared maps for sched_ext.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_always_max_xcoord.skel.h"
#include "cleanup_struct_ops.h"
#include "shared_maps.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
	exiting = true;
}

static void print_gpu_state(struct prefetch_always_max_xcoord_bpf *skel)
{
	int fd = bpf_map__fd(skel->maps.gpu_state_map);
	__u32 key = 0, next_key;
	struct gpu_pid_state state;

	while (bpf_map_get_next_key(fd, &key, &next_key) == 0) {
		if (bpf_map_lookup_elem(fd, &next_key, &state) == 0) {
			printf("PID %-8u  fault_rate=%-5llu  fault_cnt=%-5llu  "
			       "evict_cnt=%-8llu  used_cnt=%-8llu  thrashing=%s\n",
			       next_key,
			       (unsigned long long)state.fault_rate,
			       (unsigned long long)state.fault_count,
			       (unsigned long long)state.eviction_count,
			       (unsigned long long)state.used_count,
			       state.is_thrashing ? "YES" : "no");
		}
		key = next_key;
	}
}

int main(int argc, char **argv) {
	struct prefetch_always_max_xcoord_bpf *skel;
	struct bpf_link *link;
	int err;

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);

	cleanup_old_struct_ops();

	skel = prefetch_always_max_xcoord_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = prefetch_always_max_xcoord_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* Pin gpu_state_map for sched_ext to read */
	int gpu_state_fd = bpf_map__fd(skel->maps.gpu_state_map);
	unlink(XCOORD_GPU_STATE_PIN);
	err = bpf_obj_pin(gpu_state_fd, XCOORD_GPU_STATE_PIN);
	if (err) {
		fprintf(stderr, "Failed to pin gpu_state_map at %s: %s\n",
			XCOORD_GPU_STATE_PIN, strerror(errno));
		fprintf(stderr, "Make sure /sys/fs/bpf/ is mounted (bpffs)\n");
		goto cleanup;
	}
	printf("Pinned gpu_state_map at %s\n", XCOORD_GPU_STATE_PIN);

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
	link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_always_max_xcoord);
	if (!link) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n",
			strerror(-err), err);
		unlink(XCOORD_GPU_STATE_PIN);
		unlink(XCOORD_UVM_WORKERS_PIN);
		goto cleanup;
	}

	printf("xCoord GPU-side loaded: always_max prefetch + cycle_moe eviction + shared state\n");
	printf("  Shared maps: %s, %s\n", XCOORD_GPU_STATE_PIN, XCOORD_UVM_WORKERS_PIN);
	printf("\nPress Ctrl-C to exit...\n");

	while (!exiting) {
		sleep(5);
		print_gpu_state(skel);
	}

	printf("\nDetaching...\n");
	print_gpu_state(skel);
	bpf_link__destroy(link);

	/* Unpin shared maps on exit */
	unlink(XCOORD_GPU_STATE_PIN);
	unlink(XCOORD_UVM_WORKERS_PIN);
	printf("Unpinned shared maps\n");

cleanup:
	prefetch_always_max_xcoord_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
