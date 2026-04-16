/* Template-Aware Prefetch + Belady OPT Eviction — Userspace Loader
 *
 * Loads NVBit profiling data (or analytical model data) into BPF maps,
 * then attaches the template_belady struct_ops program.
 *
 * Usage:
 *   ./prefetch_template_belady [--profile profiling_data/kernel_templates.json]
 *                               [--layers 36] [--protect-distance 3]
 *
 * Without --profile, uses analytical defaults for 120B GPT-OSS MoE model.
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

#include "prefetch_template_belady.skel.h"
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
    __u32 t1_freq_threshold;
    __u32 protect_distance;
    __u64 model_va_start;
    __u64 model_va_end;
};

/* Parse a hex string "0x..." to unsigned long long */
static unsigned long long parse_hex_value(const char *line, const char *key)
{
    char *p = strstr(line, key);
    if (!p) return 0;
    char *q = strchr(p + strlen(key), '\"');
    if (!q) return 0;
    q++;
    return strtoull(q, NULL, 16);
}

/* Load boundary_vas array from equal-count JSON into BPF array map.
 * Also returns model_va_start and model_va_end. */
static int load_boundaries_from_json(int boundary_fd, int *out_num_layers,
                                     unsigned long long *out_va_start,
                                     unsigned long long *out_va_end,
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
        /* Parse model_va_start */
        if (strstr(line, "\"model_va_start\"")) {
            *out_va_start = parse_hex_value(line, "\"model_va_start\"");
            continue;
        }
        /* Parse model_va_end */
        if (strstr(line, "\"model_va_end\"")) {
            *out_va_end = parse_hex_value(line, "\"model_va_end\"");
            continue;
        }
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
            /* Parse "0x..." entries */
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

/* Legacy: load per-chunk VA→layer entries from layer_va_ranges.json into hash map */
static int load_va_layer_map_from_json(int map_fd, const char *json_path)
{
    FILE *f = fopen(json_path, "r");
    if (!f) return -1;

    char line[4096];
    int entries = 0;
    __u32 current_layer_id = 0xFFFFFFFF;
    unsigned long long current_va_start = 0, current_va_end = 0;

    while (fgets(line, sizeof(line), f)) {
        char *p;
        p = strstr(line, "\"layer_id\":");
        if (p) { current_layer_id = (__u32)strtoul(p + 12, NULL, 10); continue; }
        p = strstr(line, "\"va_start\":");
        if (p) {
            char *q = strchr(p + 11, '\"'); if (q) { q++; char *end = strchr(q, '\"'); if (end) *end = '\0'; current_va_start = strtoull(q, NULL, 16); }
            continue;
        }
        p = strstr(line, "\"va_end\":");
        if (p) {
            char *q = strchr(p + 9, '\"'); if (q) { q++; char *end = strchr(q, '\"'); if (end) *end = '\0'; current_va_end = strtoull(q, NULL, 16); }
            if (current_layer_id != 0xFFFFFFFF && current_va_start && current_va_end) {
                for (unsigned long long va = current_va_start; va < current_va_end; va += (1ULL << 21)) {
                    __u32 key = (__u32)(va >> 21), value = current_layer_id;
                    bpf_map_update_elem(map_fd, &key, &value, BPF_ANY);
                    entries++;
                }
                current_layer_id = 0xFFFFFFFF; current_va_start = 0; current_va_end = 0;
            }
            continue;
        }
    }
    fclose(f);
    return entries;
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("\nOptions:\n");
    printf("  --profile PATH    Path to layer_va_ranges.json from process_ws_trace.py\n");
    printf("  --layers N        Number of model layers (default: 36)\n");
    printf("  --protect N       Protect layers within N of current (default: 3)\n");
    printf("  --t1-threshold N  T1 frequency threshold (default: 3)\n");
    printf("  -h, --help        Show this help\n");
}

int main(int argc, char **argv) {
    struct prefetch_template_belady_bpf *skel;
    struct bpf_link *link;
    int err;

    /* Options */
    const char *profile_path = NULL;
    __u32 num_layers = 36;
    __u32 protect_distance = 3;
    __u32 t1_threshold = 3;

    static struct option long_options[] = {
        {"profile",      required_argument, 0, 'p'},
        {"layers",       required_argument, 0, 'l'},
        {"protect",      required_argument, 0, 'd'},
        {"t1-threshold", required_argument, 0, 't'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "p:l:d:t:h", long_options, NULL)) != -1) {
        switch (c) {
        case 'p': profile_path = optarg; break;
        case 'l': num_layers = (__u32)atoi(optarg); break;
        case 'd': protect_distance = (__u32)atoi(optarg); break;
        case 't': t1_threshold = (__u32)atoi(optarg); break;
        case 'h': print_usage(argv[0]); return 0;
        default: print_usage(argv[0]); return 1;
        }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);
    cleanup_old_struct_ops();

    skel = prefetch_template_belady_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_template_belady_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Load VA → layer mapping from profile data */
    unsigned long long model_va_start = 0, model_va_end = 0;
    if (profile_path) {
        /* Try loading as boundary-based format first (equal-count model) */
        int boundary_fd = bpf_map__fd(skel->maps.layer_boundaries);
        int detected_layers = 0;
        int entries = load_boundaries_from_json(boundary_fd, &detected_layers,
                                                &model_va_start, &model_va_end,
                                                profile_path);
        if (entries > 0) {
            printf("Loaded %d layer boundaries from %s\n", entries, profile_path);
            if (detected_layers > 0)
                num_layers = detected_layers;
        } else {
            /* Fallback: load as per-chunk hash map format */
            int va_layer_fd = bpf_map__fd(skel->maps.va_to_layer_map);
            entries = load_va_layer_map_from_json(va_layer_fd, profile_path);
            if (entries < 0) {
                fprintf(stderr, "Warning: Failed to load profile data, running without VA mapping\n");
            } else {
                printf("Loaded %d VA→layer hash entries from %s\n", entries, profile_path);
            }
        }
    } else {
        printf("No profile data — running with runtime layer learning only\n");
        printf("  (Use --profile to load layer_va_ranges_equal_count.json)\n");
    }

    /* Populate config map */
    struct config cfg = {
        .num_layers = num_layers,
        .t1_freq_threshold = t1_threshold,
        .protect_distance = protect_distance,
        .model_va_start = model_va_start,
        .model_va_end = model_va_end,
    };
    __u32 zero = 0;
    int config_fd = bpf_map__fd(skel->maps.config_map);
    bpf_map_update_elem(config_fd, &zero, &cfg, BPF_ANY);

    printf("Config: layers=%u, protect_distance=%u, t1_threshold=%u\n",
           num_layers, protect_distance, t1_threshold);
    if (model_va_start)
        printf("Model VA range: 0x%llx - 0x%llx\n", model_va_start, model_va_end);

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_template_belady);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("\nLoaded: template-aware prefetch + Belady eviction\n");
    printf("  Prefetch: always_max (full VA block)\n");
    printf("  Eviction: T1 protect (freq >= %u) + Belady cycle distance\n", t1_threshold);
    printf("  Layer detection: %s\n",
           profile_path ? "NVBit VA mapping + runtime tracking" : "runtime VA tracking only");
    printf("Press Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_template_belady_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
