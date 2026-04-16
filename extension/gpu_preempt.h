// SPDX-License-Identifier: GPL-2.0
/*
 * gpu_preempt.h - Reusable GPU TSG preempt mechanism
 *
 * Zero kernel modification needed. Handle capture uses ioctl interception,
 * preempt uses CUDA's own fd (same nvfp → passes RM security check).
 *
 * Usage:
 *   1. Load BPF probes (kprobe/kretprobe on nvidia_unlocked_ioctl)
 *   2. Initialize CUDA (creates TSGs captured by BPF)
 *   3. Read captured TSGs from BPF map
 *   4. Find CUDA's nvidia fd via gp_find_cuda_fd()
 *   5. Use gp_preempt() / gp_set_timeslice() with that fd
 */

#ifndef GPU_PREEMPT_H
#define GPU_PREEMPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <cuda.h>

/* ─── NVIDIA ioctl constants ─── */
#define GP_NV_IOCTL_MAGIC       'F'
#define GP_NV_IOCTL_BASE        200
#define GP_NV_ESC_RM_CONTROL    0x2A
#define GP_NV_ESC_IOCTL_XFER_CMD (GP_NV_IOCTL_BASE + 11)
#define GP_NVA06C_CTRL_CMD_PREEMPT       0xa06c0105
#define GP_NVA06C_CTRL_CMD_SET_TIMESLICE 0xa06c0103

/* ─── NVIDIA ioctl structs ─── */
typedef struct {
	uint32_t cmd;
	uint32_t size;
	void    *ptr __attribute__((aligned(8)));
} gp_nv_ioctl_xfer_t;

typedef struct {
	uint32_t hClient;
	uint32_t hObject;
	uint32_t cmd;
	uint32_t flags;
	void    *params __attribute__((aligned(8)));
	uint32_t paramsSize;
	uint32_t status;
} gp_NVOS54_PARAMETERS;

typedef struct {
	uint8_t  bWait;
	uint8_t  bManualTimeout;
	uint32_t timeoutUs;
} gp_preempt_params_t;

typedef struct {
	uint64_t timesliceUs;
} gp_timeslice_params_t;

/* ─── TSG entry (must match BPF side) ─── */
struct tsg_entry {
	uint32_t hClient;
	uint32_t hTsg;
	uint32_t engine_type;
	uint32_t pid;
	uint64_t tsg_id;
};

/* ─── Helpers ─── */

static inline uint64_t gp_get_time_us(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static inline const char *gp_engine_str(uint32_t e)
{
	switch (e) {
	case 1:  return "GR";
	case 13: return "CE";
	case 14: return "CE2";
	default: return "?";
	}
}

#define GP_CHECK_CUDA(call) do { \
	CUresult _err = (call); \
	if (_err != CUDA_SUCCESS) { \
		const char *_errStr; \
		cuGetErrorString(_err, &_errStr); \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", \
			__FILE__, __LINE__, _errStr); \
		exit(1); \
	} \
} while (0)

/* ─── ioctl RM control ─── */

static inline int gp_rm_control(int fd, uint32_t hClient, uint32_t hObject,
				uint32_t cmd, void *params, uint32_t paramsSize)
{
	gp_NVOS54_PARAMETERS ctrl;
	gp_nv_ioctl_xfer_t xfer;
	int ret;

	memset(&ctrl, 0, sizeof(ctrl));
	ctrl.hClient = hClient;
	ctrl.hObject = hObject;
	ctrl.cmd = cmd;
	ctrl.flags = 0;
	ctrl.params = params;
	ctrl.paramsSize = paramsSize;
	ctrl.status = 0;

	memset(&xfer, 0, sizeof(xfer));
	xfer.cmd = GP_NV_ESC_RM_CONTROL;
	xfer.size = sizeof(ctrl);
	xfer.ptr = &ctrl;

	ret = ioctl(fd, _IOWR(GP_NV_IOCTL_MAGIC, GP_NV_ESC_IOCTL_XFER_CMD,
			       gp_nv_ioctl_xfer_t), &xfer);
	if (ret < 0)
		return -errno;
	return ctrl.status;
}

static inline int gp_preempt(int fd, uint32_t hClient, uint32_t hTsg)
{
	gp_preempt_params_t params = {};
	params.bWait = 1;
	params.bManualTimeout = 1;
	params.timeoutUs = 100000;
	return gp_rm_control(fd, hClient, hTsg,
			     GP_NVA06C_CTRL_CMD_PREEMPT,
			     &params, sizeof(params));
}

static inline int gp_set_timeslice(int fd, uint32_t hClient, uint32_t hTsg,
				   uint64_t us)
{
	gp_timeslice_params_t params = {};
	params.timesliceUs = us;
	return gp_rm_control(fd, hClient, hTsg,
			     GP_NVA06C_CTRL_CMD_SET_TIMESLICE,
			     &params, sizeof(params));
}

/* ─── Find CUDA's nvidia fd ─── */

static inline int gp_find_cuda_fd(uint32_t hClient, uint32_t hTsg, int verbose)
{
	char path[64], link[256];

	if (verbose)
		printf("  Scanning /proc/self/fd for CUDA's nvidia fd...\n");

	for (int fd = 3; fd < 1024; fd++) {
		snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
		ssize_t len = readlink(path, link, sizeof(link) - 1);
		if (len < 0)
			continue;
		link[len] = '\0';

		if (strncmp(link, "/dev/nvidia", 11) != 0)
			continue;

		int status = gp_preempt(fd, hClient, hTsg);
		if (verbose)
			printf("    fd=%d (%s): status=%d\n", fd, link, status);

		if (status == 0) {
			if (verbose)
				printf("  Found CUDA's fd=%d (%s)\n", fd, link);
			return fd;
		}
	}

	return -1;
}

/* ─── PTX kernel (sm_50, JIT-compiled to target GPU) ─── */

static const char *gp_ptx_source =
	".version 7.0\n"
	".target sm_50\n"
	".address_size 64\n"
	"\n"
	".visible .entry busy_loop(\n"
	"    .param .u64 param_iterations,\n"
	"    .param .u64 param_output\n"
	")\n"
	"{\n"
	"    .reg .u64 %rd<5>;\n"
	"    .reg .u32 %r<5>;\n"
	"    .reg .pred %p1, %p2;\n"
	"\n"
	"    ld.param.u64 %rd1, [param_iterations];\n"
	"    ld.param.u64 %rd3, [param_output];\n"
	"    mov.u64 %rd2, 0;\n"
	"\n"
	"loop:\n"
	"    add.u64 %rd2, %rd2, 1;\n"
	"    setp.lt.u64 %p1, %rd2, %rd1;\n"
	"    @%p1 bra loop;\n"
	"\n"
	"    mov.u32 %r1, %ctaid.x;\n"
	"    mov.u32 %r2, %ntid.x;\n"
	"    mov.u32 %r3, %tid.x;\n"
	"    mad.lo.u32 %r4, %r1, %r2, %r3;\n"
	"    setp.eq.u32 %p2, %r4, 0;\n"
	"    @%p2 st.global.u64 [%rd3], %rd2;\n"
	"\n"
	"    ret;\n"
	"}\n";

/* ─── GPU worker ─── */

#define GP_MAX_SAMPLES 64

struct gp_worker {
	CUcontext       cuda_ctx;
	uint64_t        iterations;
	volatile int   *running;
	volatile int    ready;
	pthread_t       thread;
	pthread_mutex_t lock;
	/* stats */
	uint64_t kernel_count;
	uint64_t total_time_us;
	uint64_t last_kernel_us;
	uint64_t samples[GP_MAX_SAMPLES];
	uint32_t sample_idx;
	int      recording;
};

static inline void gp_worker_init(struct gp_worker *w, CUcontext ctx,
				  uint64_t iterations, volatile int *running)
{
	memset(w, 0, sizeof(*w));
	w->cuda_ctx = ctx;
	w->iterations = iterations;
	w->running = running;
	pthread_mutex_init(&w->lock, NULL);
}

static void *gp_worker_fn(void *arg)
{
	struct gp_worker *w = arg;
	CUmodule module;
	CUfunction kernel;
	CUdeviceptr d_output;

	GP_CHECK_CUDA(cuCtxSetCurrent(w->cuda_ctx));
	GP_CHECK_CUDA(cuModuleLoadData(&module, gp_ptx_source));
	GP_CHECK_CUDA(cuModuleGetFunction(&kernel, module, "busy_loop"));
	GP_CHECK_CUDA(cuMemAlloc(&d_output, sizeof(uint64_t)));

	w->ready = 1;

	while (*(w->running)) {
		uint64_t iters = w->iterations;
		void *args[] = { &iters, &d_output };
		uint64_t start = gp_get_time_us();

		CUresult res = cuLaunchKernel(kernel,
					      128, 1, 1, 256, 1, 1,
					      0, 0, args, NULL);
		if (res != CUDA_SUCCESS)
			break;

		cuCtxSynchronize();
		uint64_t dur = gp_get_time_us() - start;

		pthread_mutex_lock(&w->lock);
		w->kernel_count++;
		w->total_time_us += dur;
		w->last_kernel_us = dur;
		if (w->recording && w->sample_idx < GP_MAX_SAMPLES)
			w->samples[w->sample_idx++] = dur;
		pthread_mutex_unlock(&w->lock);
	}

	cuMemFree(d_output);
	cuModuleUnload(module);
	return NULL;
}

static inline int gp_worker_start(struct gp_worker *w)
{
	return pthread_create(&w->thread, NULL, gp_worker_fn, w);
}

static inline void gp_worker_wait_ready(struct gp_worker *w)
{
	while (!w->ready)
		usleep(10000);
}

static inline void gp_worker_join(struct gp_worker *w)
{
	pthread_join(w->thread, NULL);
}

/* CUDA context warmup (ensures all TSGs are created) */
static inline void gp_cuda_warmup(CUcontext ctx)
{
	CUmodule m;
	CUfunction f;
	CUdeviceptr d_tmp;
	uint64_t iters = 1;

	GP_CHECK_CUDA(cuCtxSetCurrent(ctx));
	GP_CHECK_CUDA(cuMemAlloc(&d_tmp, sizeof(uint64_t)));
	GP_CHECK_CUDA(cuModuleLoadData(&m, gp_ptx_source));
	GP_CHECK_CUDA(cuModuleGetFunction(&f, m, "busy_loop"));

	void *args[] = { &iters, &d_tmp };
	GP_CHECK_CUDA(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL));
	GP_CHECK_CUDA(cuCtxSynchronize());
	cuModuleUnload(m);
	cuMemFree(d_tmp);
}

#endif /* GPU_PREEMPT_H */
