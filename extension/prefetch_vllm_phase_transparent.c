/* vLLM Transparent Uprobe Phase Detection - Loader
 *
 * Attaches uprobes on paged_attention_v1()/paged_attention_v2() in vLLM's
 * _C.abi3.so to detect decode batches without modifying vLLM source.
 *
 * Usage:
 *   sudo ./prefetch_vllm_phase_transparent [path-to-_C.abi3.so] [--paged-attn-v2]
 */
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_vllm_phase_transparent.skel.h"
#include "cleanup_struct_ops.h"

#define DEFAULT_VLLM_C_ABI3 \
	"/home/yunwei37/workspace/gpu/gpu_ext/workloads/vllm/vllm/vllm/_C.abi3.so"

#define PREFILL_TIMEOUT_FAULTS 500
#define PHASE_PREFILL 1
#define PHASE_DECODE  2

static const char *const PAGED_ATTN_V1 = "paged_attention_v1";
static const char *const PAGED_ATTN_V1_MANGLED =
	"_Z18paged_attention_v1RN2at6TensorES1_S1_S1_ldS1_S1_llRKSt8optionalIS0_ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES1_S1_lllll";
static const char *const PAGED_ATTN_V2 = "paged_attention_v2";
static const char *const PAGED_ATTN_V2_MANGLED =
	"_Z18paged_attention_v2RN2at6TensorES1_S1_S1_S1_S1_S1_ldS1_S1_llRKSt8optionalIS0_ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES1_S1_lllll";

static volatile bool exiting;

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
	return vfprintf(stderr, format, args);
}

static void handle_signal(int sig)
{
	exiting = true;
}

static void usage(const char *prog)
{
	fprintf(stderr, "Usage: %s [path-to-_C.abi3.so] [--paged-attn-v2]\n", prog);
	fprintf(stderr, "Default path: %s\n", DEFAULT_VLLM_C_ABI3);
}

static int read_u32_map(int fd, __u32 key, __u32 *value)
{
	return bpf_map_lookup_elem(fd, &key, value);
}

static int read_u64_map(int fd, __u32 key, __u64 *value)
{
	return bpf_map_lookup_elem(fd, &key, value);
}

static struct bpf_link *attach_uprobe_with_fallback(struct bpf_program *prog,
						    const char *binary_path,
						    const char *func_name,
						    const char *fallback_symbol,
						    const char **attached_symbol)
{
	struct bpf_link *link;
	int err;

	LIBBPF_OPTS(bpf_uprobe_opts, primary_opts,
		.func_name = func_name,
		.retprobe = false,
	);
	link = bpf_program__attach_uprobe_opts(prog, -1, binary_path, 0, &primary_opts);
	err = libbpf_get_error(link);
	if (!err) {
		if (attached_symbol)
			*attached_symbol = func_name;
		return link;
	}

	link = NULL;
	if (!fallback_symbol || strcmp(fallback_symbol, func_name) == 0) {
		errno = -err;
		return NULL;
	}

	LIBBPF_OPTS(bpf_uprobe_opts, fallback_opts,
		.func_name = fallback_symbol,
		.retprobe = false,
	);
	link = bpf_program__attach_uprobe_opts(prog, -1, binary_path, 0, &fallback_opts);
	err = libbpf_get_error(link);
	if (!err) {
		if (attached_symbol)
			*attached_symbol = fallback_symbol;
		return link;
	}

	errno = -err;
	return NULL;
}

int main(int argc, char **argv)
{
	struct prefetch_vllm_phase_transparent_bpf *skel = NULL;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	struct bpf_link *link_uprobe_v1 = NULL;
	struct bpf_link *link_uprobe_v2 = NULL;
	const char *lib_path = DEFAULT_VLLM_C_ABI3;
	const char *attached_v1 = NULL;
	const char *attached_v2 = NULL;
	bool attach_v2 = false;
	int err = 0;
	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--paged-attn-v2") == 0) {
			attach_v2 = true;
			continue;
		}
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			usage(argv[0]);
			return 0;
		}
		if (argv[i][0] == '-') {
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			usage(argv[0]);
			return 1;
		}
		lib_path = argv[i];
	}

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_vllm_phase_transparent_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = prefetch_vllm_phase_transparent_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	int phase_fd = bpf_map__fd(skel->maps.phase_map);
	int faults_fd = bpf_map__fd(skel->maps.faults_since_decode);
	if (phase_fd >= 0) {
		__u32 key = 0;
		__u32 phase = PHASE_PREFILL;
		bpf_map_update_elem(phase_fd, &key, &phase, BPF_ANY);
	}
	if (faults_fd >= 0) {
		__u32 key = 0;
		__u64 faults = 0;
		bpf_map_update_elem(faults_fd, &key, &faults, BPF_ANY);
	}

	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_vllm_phase_transparent);
	err = libbpf_get_error(link_struct_ops);
	if (err) {
		link_struct_ops = NULL;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("struct_ops attached (vllm_phase_transparent)\n");

	link_kprobe = bpf_program__attach(skel->progs.capture_va_block);
	err = libbpf_get_error(link_kprobe);
	if (err) {
		link_kprobe = NULL;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("kprobe attached (va_block capture + phase timeout)\n");

	link_uprobe_v1 = attach_uprobe_with_fallback(
		skel->progs.paged_attention_v1_decode,
		lib_path,
		PAGED_ATTN_V1,
		PAGED_ATTN_V1_MANGLED,
		&attached_v1);
	if (!link_uprobe_v1) {
		err = -errno;
		fprintf(stderr, "Failed to attach uprobe on %s:%s: %s (%d)\n",
			lib_path, PAGED_ATTN_V1, strerror(errno), err);
		goto cleanup;
	}
	printf("uprobe attached: %s -> DECODE (%s)\n", PAGED_ATTN_V1, attached_v1);

	if (attach_v2) {
		link_uprobe_v2 = attach_uprobe_with_fallback(
			skel->progs.paged_attention_v2_decode,
			lib_path,
			PAGED_ATTN_V2,
			PAGED_ATTN_V2_MANGLED,
			&attached_v2);
		if (!link_uprobe_v2) {
			err = -errno;
			fprintf(stderr, "Failed to attach uprobe on %s:%s: %s (%d)\n",
				lib_path, PAGED_ATTN_V2, strerror(errno), err);
			goto cleanup;
		}
		printf("uprobe attached: %s -> DECODE (%s)\n", PAGED_ATTN_V2, attached_v2);
	}

	printf("\n=== vLLM Transparent Uprobe Phase Detection ===\n");
	printf("  Library: %s\n", lib_path);
	printf("  Decode marker: %s%s\n", PAGED_ATTN_V1, attach_v2 ? " + paged_attention_v2" : "");
	printf("  Timeout back to PREFILL: %d faults without paged attention\n",
	       PREFILL_TIMEOUT_FAULTS);
	printf("  Prefill: always_max + direction-aware cross-block\n");
	printf("  Decode: always_max only (cross-block OFF)\n");
	printf("  Eviction: cycle_moe (T1 protect + DEFAULT non-T1)\n");
	printf("Press Ctrl-C to exit...\n\n");

	int stats_fd = bpf_map__fd(skel->maps.vtp_stats);

	while (!exiting) {
		__u32 phase_key = 0;
		__u32 phase_val = PHASE_PREFILL;
		__u64 faults_val = 0;
		__u64 stat_v1 = 0;
		__u64 stat_v2 = 0;
		__u64 stat_timeout = 0;
		__u64 stat_prefetch = 0;
		__u64 stat_xb = 0;
		__u64 stat_decode_skip = 0;
		__u64 stat_wq = 0;
		__u64 stat_migrate = 0;
		__u32 stat_key;

		sleep(2);

		if (phase_fd >= 0)
			read_u32_map(phase_fd, phase_key, &phase_val);
		if (faults_fd >= 0)
			read_u64_map(faults_fd, phase_key, &faults_val);
		if (stats_fd >= 0) {
			stat_key = 10; read_u64_map(stats_fd, stat_key, &stat_v1);
			stat_key = 11; read_u64_map(stats_fd, stat_key, &stat_v2);
			stat_key = 12; read_u64_map(stats_fd, stat_key, &stat_timeout);
			stat_key = 1; read_u64_map(stats_fd, stat_key, &stat_prefetch);
			stat_key = 13; read_u64_map(stats_fd, stat_key, &stat_xb);
			stat_key = 14; read_u64_map(stats_fd, stat_key, &stat_decode_skip);
			stat_key = 2; read_u64_map(stats_fd, stat_key, &stat_wq);
			stat_key = 4; read_u64_map(stats_fd, stat_key, &stat_migrate);
		}

		printf("Phase: %s | uprobe=%llu",
		       phase_val == PHASE_DECODE ? "DECODE" : "PREFILL",
		       (unsigned long long)(stat_v1 + stat_v2));
		printf(" v1=%llu", (unsigned long long)stat_v1);
		if (attach_v2)
			printf(" v2=%llu", (unsigned long long)stat_v2);
		printf(" timeout_prefill=%llu", (unsigned long long)stat_timeout);
		printf(" faults_since_decode=%llu", (unsigned long long)faults_val);
		printf(" prefetch=%llu", (unsigned long long)stat_prefetch);
		printf(" xb_prefill=%llu", (unsigned long long)stat_xb);
		printf(" decode_skip=%llu", (unsigned long long)stat_decode_skip);
		printf(" wq_sched=%llu", (unsigned long long)stat_wq);
		printf(" migrate_ok=%llu\n", (unsigned long long)stat_migrate);
	}

	printf("\n=== Final vLLM Transparent Phase Stats ===\n");
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
			"phase -> DECODE (paged_attention_v1)",
			"phase -> DECODE (paged_attention_v2)",
			"phase -> PREFILL (fault timeout)",
			"prefill XB (cross-block during prefill)",
			"decode skip (XB off during decode)",
			"decode faults observed",
		};
		__u32 stat_key;

		for (stat_key = 0; stat_key < 16; stat_key++) {
			__u64 value = 0;

			read_u64_map(stats_fd, stat_key, &value);
			printf("  %-42s %llu\n", stat_names[stat_key],
			       (unsigned long long)value);
		}
	}

	if (phase_fd >= 0) {
		__u32 phase_key = 0;
		__u32 phase_val = PHASE_PREFILL;

		if (read_u32_map(phase_fd, phase_key, &phase_val) == 0)
			printf("\n  Final phase: %s\n",
			       phase_val == PHASE_DECODE ? "DECODE" : "PREFILL");
	}

	if (faults_fd >= 0) {
		__u32 key = 0;
		__u64 faults = 0;

		if (read_u64_map(faults_fd, key, &faults) == 0)
			printf("  Faults since decode: %llu\n", (unsigned long long)faults);
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_uprobe_v2)
		bpf_link__destroy(link_uprobe_v2);
	if (link_uprobe_v1)
		bpf_link__destroy(link_uprobe_v1);
	if (link_kprobe)
		bpf_link__destroy(link_kprobe);
	if (link_struct_ops)
		bpf_link__destroy(link_struct_ops);
	prefetch_vllm_phase_transparent_bpf__destroy(skel);
	return err < 0 ? -err : err;
}
