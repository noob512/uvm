/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Proactive Layer Migration — Userspace Loader
 *
 * Loads layer boundary data and attaches the proactive_layer BPF program.
 * Periodically prints stats (proactive prefetches scheduled, completed, etc.)
 *
 * Usage:
 *   ./prefetch_proactive_layer [--profile layer_va_ranges_equal_count.json]
 *                              [--layers 36] [--prefetch-kb 4096] [--ahead 1]
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

#include "prefetch_proactive_layer.skel.h"
#include "cleanup_struct_ops.h"

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

struct config {
    __u32 num_layers;
    __u32 prefetch_ahead;
    __u64 prefetch_bytes;
};

/* Load boundary_vas from equal-count JSON into BPF array map */
static int load_boundaries_from_json(int boundary_fd, int *out_num_layers,
                                     const char *json_path)
{
    FILE *f = fopen(json_path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s: %s\n", json_path, strerror(errno));
        return -1;
    }

    char line[4096];
    int entries = 0;
    int in_boundary_array = 0;

    while (fgets(line, sizeof(line), f)) {
        /* Parse num_layers */
        char *p = strstr(line, "\"num_layers\":");
        if (p) {
            *out_num_layers = (int)strtol(p + 14, NULL, 10);
            continue;
        }
        /* Detect boundary_vas array */
        if (strstr(line, "\"boundary_vas\"")) {
            in_boundary_array = 1;
            continue;
        }
        if (in_boundary_array) {
            if (strchr(line, ']')) {
                in_boundary_array = 0;
                continue;
            }
            char *q = strchr(line, '\"');
            if (q) {
                q++;
                char *end = strchr(q, '\"');
                if (end) *end = '\0';
                __u64 boundary_va = strtoull(q, NULL, 16);
                __u32 key = entries;
                bpf_map_update_elem(boundary_fd, &key, &boundary_va, BPF_ANY);
                entries++;
            }
        }
    }

    fclose(f);
    return entries;
}

static const char *stat_names[] = {
    "kprobe_fires",
    "prefetch_hooks",
    "layer_transitions",
    "wq_scheduled",
    "wq_callbacks",
    "migrate_ok",
    "migrate_fail",
    "token_boundaries",
    "dedup_skip",
    NULL,
};

static void print_stats(int stats_fd)
{
    printf("\n--- Proactive Layer Migration Stats ---\n");
    for (__u32 i = 0; stat_names[i]; i++) {
        __u64 val = 0;
        bpf_map_lookup_elem(stats_fd, &i, &val);
        printf("  %-20s %llu\n", stat_names[i], val);
    }
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  --profile PATH    Path to layer_va_ranges_equal_count.json\n");
    printf("  --layers N        Number of model layers (default: 36)\n");
    printf("  --prefetch-kb N   KB to pre-migrate per layer transition (default: 4096)\n");
    printf("  --ahead N         Layers ahead to pre-migrate (1-3, default: 1)\n");
    printf("  -h, --help        Show this help\n");
}

int main(int argc, char **argv) {
    struct prefetch_proactive_layer_bpf *skel;
    struct bpf_link *link = NULL;
    struct bpf_link *kprobe_link = NULL;
    int err;

    /* Options */
    const char *profile_path = NULL;
    __u32 num_layers = 36;
    __u32 prefetch_ahead = 1;
    __u64 prefetch_kb = 4096;  /* 4 MB default */

    static struct option long_options[] = {
        {"profile",      required_argument, 0, 'p'},
        {"layers",       required_argument, 0, 'l'},
        {"prefetch-kb",  required_argument, 0, 'k'},
        {"ahead",        required_argument, 0, 'a'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "p:l:k:a:h", long_options, NULL)) != -1) {
        switch (c) {
        case 'p': profile_path = optarg; break;
        case 'l': num_layers = (__u32)atoi(optarg); break;
        case 'k': prefetch_kb = (__u64)atoll(optarg); break;
        case 'a': prefetch_ahead = (__u32)atoi(optarg); break;
        case 'h': print_usage(argv[0]); return 0;
        default: print_usage(argv[0]); return 1;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);
    cleanup_old_struct_ops();

    skel = prefetch_proactive_layer_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_proactive_layer_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Load layer boundaries from profile data */
    if (profile_path) {
        int boundary_fd = bpf_map__fd(skel->maps.layer_boundaries);
        int detected_layers = 0;
        int entries = load_boundaries_from_json(boundary_fd, &detected_layers,
                                                profile_path);
        if (entries > 0) {
            printf("Loaded %d layer boundaries from %s\n", entries, profile_path);
            if (detected_layers > 0)
                num_layers = detected_layers;
        } else {
            fprintf(stderr, "Warning: Failed to load boundaries, proactive migration disabled\n");
        }
    } else {
        fprintf(stderr, "Warning: No --profile specified, proactive migration will not work\n");
        fprintf(stderr, "  Use: --profile results/msched_trace/layer_va_ranges_equal_count.json\n");
    }

    /* Populate config map */
    struct config cfg = {
        .num_layers = num_layers,
        .prefetch_ahead = prefetch_ahead,
        .prefetch_bytes = prefetch_kb * 1024,
    };
    __u32 zero = 0;
    int config_fd = bpf_map__fd(skel->maps.config_map);
    bpf_map_update_elem(config_fd, &zero, &cfg, BPF_ANY);

    /* Attach kprobe for va_block context capture */
    kprobe_link = bpf_program__attach(skel->progs.capture_va_block);
    if (!kprobe_link) {
        err = -errno;
        fprintf(stderr, "Failed to attach kprobe: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    /* Attach struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_proactive_layer);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("\n=== Proactive Layer Migration ===\n");
    printf("  Prefetch: always_max (full VA block)\n");
    printf("  Eviction: cycle_moe (T1 protection, freq >= %d)\n", 3);
    printf("  Proactive: pre-migrate %llu KB, %u layer(s) ahead\n",
           prefetch_kb, prefetch_ahead);
    printf("  Layers: %u\n", num_layers);
    printf("  Profile: %s\n", profile_path ? profile_path : "(none)");
    printf("Press Ctrl-C to exit...\n\n");

    int stats_fd = bpf_map__fd(skel->maps.stats);
    int interval = 0;

    while (!exiting) {
        sleep(1);
        interval++;
        if (interval % 10 == 0) {
            print_stats(stats_fd);
        }
    }

    /* Final stats */
    print_stats(stats_fd);

    printf("\nDetaching...\n");
    bpf_link__destroy(link);
    link = NULL;
    bpf_link__destroy(kprobe_link);
    kprobe_link = NULL;

cleanup:
    if (link) bpf_link__destroy(link);
    if (kprobe_link) bpf_link__destroy(kprobe_link);
    prefetch_proactive_layer_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
