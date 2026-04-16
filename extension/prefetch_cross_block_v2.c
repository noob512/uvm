/* Cross-VA-Block Prefetch v2 (BPF Workqueue) - Loader */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_cross_block_v2.skel.h"
#include "cleanup_struct_ops.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

static const char *evict_mode_names[] = {
    "passive_mru (T1 protect + BYPASS non-T1)",
    "cycle_moe (T1 protect + DEFAULT non-T1)",
    "default_lru (all DEFAULT, kernel LRU)",
    "fifo (all BYPASS, freeze positions)",
};

int main(int argc, char **argv) {
    struct prefetch_cross_block_v2_bpf *skel;
    struct bpf_link *link;
    struct bpf_link *kprobe_link;
    int err;
    __u64 evict_mode = 1;     /* default: cycle_moe */
    __u64 prefetch_len = 0;   /* 0 = default (2MB) */
    __u64 prefetch_mode = 0;  /* 0 = direction-aware, 1 = blind adjacent */

    if (argc > 1) {
        evict_mode = atoi(argv[1]);
        if (evict_mode > 3) {
            fprintf(stderr, "Usage: %s [eviction_mode] [prefetch_kb] [prefetch_mode]\n", argv[0]);
            fprintf(stderr, "  Eviction modes: 0=passive_mru 1=cycle_moe(default) 2=default_lru 3=fifo\n");
            fprintf(stderr, "  Prefetch KB: 64, 256, 512, 2048 (default)\n");
            fprintf(stderr, "  Prefetch mode: 0=direction-aware(default) 1=blind_adjacent\n");
            return 1;
        }
    }
    if (argc > 2) {
        prefetch_len = (__u64)atoi(argv[2]) * 1024;
        if (prefetch_len == 0) {
            fprintf(stderr, "Invalid prefetch_kb: %s\n", argv[2]);
            return 1;
        }
    }
    if (argc > 3) {
        prefetch_mode = atoi(argv[3]);
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);
    cleanup_old_struct_ops();

    skel = prefetch_cross_block_v2_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_cross_block_v2_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set config: key 0 = eviction mode, key 1 = prefetch length */
    int config_fd = bpf_map__fd(skel->maps.xb_config);
    if (config_fd >= 0) {
        __u32 key = 0;
        bpf_map_update_elem(config_fd, &key, &evict_mode, 0);
        if (prefetch_len > 0) {
            key = 1;
            bpf_map_update_elem(config_fd, &key, &prefetch_len, 0);
        }
        key = 2;
        bpf_map_update_elem(config_fd, &key, &prefetch_mode, 0);
    }

    /* Attach kprobe for va_block context capture */
    kprobe_link = bpf_program__attach(skel->progs.capture_va_block);
    if (!kprobe_link) {
        err = -errno;
        fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_cross_block_v2);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        bpf_link__destroy(kprobe_link);
        goto cleanup;
    }

    printf("Loaded: cross-block v2 (bpf_wq) + always_max\n");
    printf("  Prefetch: intra-block always_max + 1 adjacent block (bpf_wq async)\n");
    printf("  Prefetch size: %s\n", prefetch_len > 0 ?
           (prefetch_len >= 1048576 ? "2MB" :
            prefetch_len >= 524288 ? "512KB" :
            prefetch_len >= 262144 ? "256KB" : "64KB") : "2MB (default)");
    printf("  Prefetch mode: %s\n", prefetch_mode == 0 ? "direction-aware" : "blind adjacent");
    printf("  Eviction: mode %llu = %s\n", evict_mode, evict_mode_names[evict_mode]);
    printf("Press Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(1);
    }

    /* Print debug stats before detaching */
    printf("\n=== Cross-block v2 Stats ===\n");
    int stats_fd = bpf_map__fd(skel->maps.xb_stats);
    if (stats_fd >= 0) {
        const char *stat_names[] = {
            "kprobe fires", "prefetch_hook fires",
            "wq scheduled", "wq callback ran",
            "migrate success", "migrate failed",
            "rate-limit skip", "wq callback skip",
            "direction skip", "dedup skip"
        };
        for (__u32 i = 0; i < 10; i++) {
            __u64 val = 0;
            bpf_map_lookup_elem(stats_fd, &i, &val);
            printf("  %s: %llu\n", stat_names[i], val);
        }
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);
    bpf_link__destroy(kprobe_link);

cleanup:
    prefetch_cross_block_v2_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
