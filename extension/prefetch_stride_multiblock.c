/* Stride-Predictive Multi-Block Prefetch - Loader
 *
 * Hardcoded cycle_moe eviction. No CLI args needed.
 * Prints stride prediction stats on exit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_stride_multiblock.skel.h"
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
    struct prefetch_stride_multiblock_bpf *skel;
    struct bpf_link *link;
    struct bpf_link *kprobe_link;
    int err;

    int opt_max_k = 6;
    if (argc >= 2)
        opt_max_k = atoi(argv[1]);

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);
    cleanup_old_struct_ops();

    skel = prefetch_stride_multiblock_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    skel->rodata->max_lookahead = opt_max_k;

    err = prefetch_stride_multiblock_bpf__load(skel);
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

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_stride_multiblock);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        bpf_link__destroy(kprobe_link);
        goto cleanup;
    }

    printf("Loaded: stride-predictive multi-block prefetch\n");
    printf("  Intra-block: always_max\n");
    printf("  Cross-block: stride-predictive, up to %d blocks lookahead\n", opt_max_k);
    printf("  Eviction: cycle_moe (T1 protect + DEFAULT non-T1)\n");
    printf("Press Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(1);
    }

    /* Print stats before detaching */
    printf("\n=== Stride Multi-Block Stats ===\n");
    int stats_fd = bpf_map__fd(skel->maps.sm_stats);
    if (stats_fd >= 0) {
        const char *stat_names[] = {
            "kprobe fires",
            "prefetch_hook fires",
            "wq scheduled",
            "wq callback ran",
            "migrate success",
            "migrate failed",
            "same-block skip",
            "wq callback skip",
            "stride match (4/4)",
            "stride miss",
            "prediction hit",
            "blocks prefetched",
        };
        for (__u32 i = 0; i < 12; i++) {
            __u64 val = 0;
            bpf_map_lookup_elem(stats_fd, &i, &val);
            printf("  %-24s %llu\n", stat_names[i], val);
        }
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);
    bpf_link__destroy(kprobe_link);

cleanup:
    prefetch_stride_multiblock_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
