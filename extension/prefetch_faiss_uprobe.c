/* FAISS Uprobe-Based Phase Detection + Cross-Block Prefetch - Loader
 *
 * Attaches uprobes on FAISS's GpuIndex::add_with_ids() and GpuIndex::search()
 * to detect BUILD vs SEARCH phases with zero per-fault overhead.
 *
 * Usage:
 *   sudo ./prefetch_faiss_uprobe <faiss_lib_path> <add_symbol> <search_symbol>
 *
 * Example:
 *   sudo ./prefetch_faiss_uprobe \
 *     /path/to/_swigfaiss.so \
 *     _ZN5faiss3gpu8GpuIndex12add_with_idsElPKfPKl \
 *     _ZNK5faiss3gpu8GpuIndex6searchElPKflPfPlPKNS_16SearchParametersE
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_faiss_uprobe.skel.h"
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
	struct prefetch_faiss_uprobe_bpf *skel;
	struct bpf_link *link_struct_ops = NULL;
	struct bpf_link *link_kprobe = NULL;
	struct bpf_link *link_uprobe_add = NULL;
	struct bpf_link *link_uprobe_search = NULL;
	int err;

	if (argc < 4) {
		fprintf(stderr, "Usage: %s <faiss_lib_path> <add_symbol> <search_symbol>\n", argv[0]);
		fprintf(stderr, "\nExample:\n");
		fprintf(stderr, "  %s /path/to/_swigfaiss.so \\\n", argv[0]);
		fprintf(stderr, "    _ZN5faiss3gpu8GpuIndex12add_with_idsElPKfPKl \\\n");
		fprintf(stderr, "    _ZNK5faiss3gpu8GpuIndex6searchElPKflPfPlPKNS_16SearchParametersE\n");
		return 1;
	}

	const char *faiss_lib_path = argv[1];
	const char *add_symbol = argv[2];
	const char *search_symbol = argv[3];

	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);

	libbpf_set_print(libbpf_print_fn);
	cleanup_old_struct_ops();

	skel = prefetch_faiss_uprobe_bpf__open();
	if (!skel) {
		fprintf(stderr, "Failed to open BPF skeleton\n");
		return 1;
	}

	err = prefetch_faiss_uprobe_bpf__load(skel);
	if (err) {
		fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	/* 1. Attach struct_ops */
	link_struct_ops = bpf_map__attach_struct_ops(skel->maps.uvm_ops_faiss_uprobe);
	if (!link_struct_ops) {
		err = -errno;
		fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("struct_ops attached (faiss_uprobe)\n");

	/* 2. Attach kprobe for va_block context capture */
	link_kprobe = bpf_program__attach(skel->progs.capture_va_block);
	if (!link_kprobe) {
		err = -errno;
		fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
		goto cleanup;
	}
	printf("kprobe attached (va_block capture)\n");

	/* 3. Attach uprobe on FAISS add_with_ids */
	LIBBPF_OPTS(bpf_uprobe_opts, uprobe_add_opts,
		.func_name = add_symbol,
		.retprobe = false,
	);
	link_uprobe_add = bpf_program__attach_uprobe_opts(
		skel->progs.faiss_add_start,
		-1,  /* all PIDs */
		faiss_lib_path,
		0,   /* offset (resolved by func_name) */
		&uprobe_add_opts
	);
	if (!link_uprobe_add) {
		err = -errno;
		fprintf(stderr, "Failed to attach uprobe on %s:%s: %s (%d)\n",
		        faiss_lib_path, add_symbol, strerror(-err), err);
		goto cleanup;
	}
	printf("uprobe attached: add -> BUILD (%s)\n", add_symbol);

	/* 4. Attach uprobe on FAISS search */
	LIBBPF_OPTS(bpf_uprobe_opts, uprobe_search_opts,
		.func_name = search_symbol,
		.retprobe = false,
	);
	link_uprobe_search = bpf_program__attach_uprobe_opts(
		skel->progs.faiss_search_start,
		-1,  /* all PIDs */
		faiss_lib_path,
		0,   /* offset (resolved by func_name) */
		&uprobe_search_opts
	);
	if (!link_uprobe_search) {
		err = -errno;
		fprintf(stderr, "Failed to attach uprobe on %s:%s: %s (%d)\n",
		        faiss_lib_path, search_symbol, strerror(-err), err);
		goto cleanup;
	}
	printf("uprobe attached: search -> SEARCH (%s)\n", search_symbol);

	printf("\n=== FAISS Uprobe Phase Detection ===\n");
	printf("  Library: %s\n", faiss_lib_path);
	printf("  Intra-block: always_max (prefetch entire VA block)\n");
	printf("  Cross-block: 2MB direction-aware, BUILD phase only\n");
	printf("  Phase detection: uprobe (zero per-fault overhead)\n");
	printf("  Eviction: default LRU (kernel handles all)\n");
	printf("Press Ctrl-C to exit...\n\n");

	int stats_fd = bpf_map__fd(skel->maps.fu_stats);

	while (!exiting) {
		sleep(2);

		/* Read phase */
		int phase_fd = bpf_map__fd(skel->maps.phase_map);
		__u32 phase_key = 0;
		__u32 phase_val = 0;
		if (phase_fd >= 0)
			bpf_map_lookup_elem(phase_fd, &phase_key, &phase_val);

		printf("Phase: %s | ",
		       phase_val == 1 ? "BUILD" :
		       phase_val == 2 ? "SEARCH" : "UNKNOWN");

		if (stats_fd >= 0) {
			__u64 val;
			__u32 k;
			/* Print selected stats inline */
			k = 10; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("uprobe_build=%llu ", (unsigned long long)val);
			k = 11; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("uprobe_search=%llu ", (unsigned long long)val);
			k = 2; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("wq_sched=%llu ", (unsigned long long)val);
			k = 4; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("migrate_ok=%llu ", (unsigned long long)val);
			k = 12; val = 0; bpf_map_lookup_elem(stats_fd, &k, &val);
			printf("search_skip=%llu", (unsigned long long)val);
		}
		printf("\n");
	}

	/* Print final stats */
	printf("\n=== Final FAISS Uprobe Stats ===\n");
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
			"phase -> BUILD",
			"phase -> SEARCH",
			"search_skip (XB off)",
			"build_prefetch (XB on)",
		};
		for (__u32 i = 0; i < 14; i++) {
			__u64 val = 0;
			bpf_map_lookup_elem(stats_fd, &i, &val);
			printf("  %s: %llu\n", stat_names[i], (unsigned long long)val);
		}
	}

	/* Print final phase */
	int phase_fd = bpf_map__fd(skel->maps.phase_map);
	if (phase_fd >= 0) {
		__u32 phase_key = 0;
		__u32 phase_val = 0;
		if (bpf_map_lookup_elem(phase_fd, &phase_key, &phase_val) == 0) {
			printf("\n  Final phase: %s\n",
			       phase_val == 1 ? "BUILD" :
			       phase_val == 2 ? "SEARCH" : "UNKNOWN");
		}
	}

	printf("\nDetaching...\n");

cleanup:
	if (link_uprobe_search) bpf_link__destroy(link_uprobe_search);
	if (link_uprobe_add) bpf_link__destroy(link_uprobe_add);
	if (link_kprobe) bpf_link__destroy(link_kprobe);
	if (link_struct_ops) bpf_link__destroy(link_struct_ops);
	prefetch_faiss_uprobe_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}
