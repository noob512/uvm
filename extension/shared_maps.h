/* SPDX-License-Identifier: GPL-2.0 */
/*
 * xCoord Shared Maps - Cross-subsystem state definitions
 *
 * Shared between gpu_ext BPF programs and sched_ext BPF programs.
 * gpu_ext writes GPU state; sched_ext reads it for scheduling decisions.
 *
 * Communication via pinned BPF maps at /sys/fs/bpf/xcoord_*
 */

#ifndef __SHARED_MAPS_H__
#define __SHARED_MAPS_H__

/* Pin paths for shared maps */
#define XCOORD_GPU_STATE_PIN "/sys/fs/bpf/xcoord_gpu_state"
#define XCOORD_UVM_WORKERS_PIN "/sys/fs/bpf/xcoord_uvm_workers"

/* Maximum tracked GPU process PIDs (set by sched_gpu_baseline loader -p flag) */
#define XCOORD_MAX_GPU_PROCS 16

/*
 * Per-PID GPU state, written by gpu_ext, read by sched_ext.
 *
 * gpu_ext updates this in chunk_activate (page fault) and eviction_prepare.
 * sched_ext reads fault_rate in enqueue() to boost UVM fault handler threads.
 */
struct gpu_pid_state {
	__u64 fault_count;      /* Faults in current window */
	__u64 fault_rate;       /* Faults/sec (updated every ~1 second) */
	__u64 eviction_count;   /* Cumulative evictions for this PID */
	__u64 used_count;       /* Cumulative chunk_used calls */
	__u64 last_update_ns;   /* Timestamp of last fault_rate update */
	__u32 is_thrashing;     /* 1 if fault_rate > thrashing threshold */
	__u32 _pad;
};

/* Fault rate thresholds for scheduling decisions */
#define XCOORD_FAULT_RATE_HIGH   1000  /* faults/sec: boost CPU priority */
#define XCOORD_FAULT_RATE_MEDIUM  100  /* faults/sec: normal priority */
#define XCOORD_THRASHING_THRESHOLD 2000 /* faults/sec: thrashing detected */

/* Maximum tracked PIDs */
#define XCOORD_MAX_PIDS 256

/* Maximum tracked UVM worker threads */
#define XCOORD_MAX_WORKERS 256

/* Worker activity timeout (5 seconds in nanoseconds) */
#define XCOORD_WORKER_TIMEOUT_NS 5000000000ULL

#endif /* __SHARED_MAPS_H__ */
