#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_serving_adaptive.skel.h"
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
    struct prefetch_serving_adaptive_bpf *skel;
    struct bpf_link *link;
    int err;

    __u64 fault_threshold = 50;
    __u64 window_ns = 10000000ULL; /* 10ms */

    if (argc > 1)
        fault_threshold = strtoull(argv[1], NULL, 10);
    if (argc > 2)
        window_ns = strtoull(argv[2], NULL, 10) * 1000000ULL; /* arg in ms */

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);
    cleanup_old_struct_ops();

    skel = prefetch_serving_adaptive_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_serving_adaptive_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    /* Set config */
    int config_fd = bpf_map__fd(skel->maps.sa_config);
    __u32 key;
    key = 0; bpf_map_update_elem(config_fd, &key, &fault_threshold, BPF_ANY);
    key = 1; bpf_map_update_elem(config_fd, &key, &window_ns, BPF_ANY);

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_serving_adaptive);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Serving-adaptive prefetch loaded!\n");
    printf("  fault_threshold=%llu faults/window\n", (unsigned long long)fault_threshold);
    printf("  window=%llu ms\n", (unsigned long long)(window_ns / 1000000));
    printf("\nPress Ctrl-C to exit...\n");

    int prefetch_fd = bpf_map__fd(skel->maps.stat_prefetch);
    int skip_fd = bpf_map__fd(skel->maps.stat_skip);

    while (!exiting) {
        sleep(5);
        __u32 k = 0;
        __u64 p = 0, s = 0;
        bpf_map_lookup_elem(prefetch_fd, &k, &p);
        bpf_map_lookup_elem(skip_fd, &k, &s);
        printf("  prefetch=%llu  skip=%llu  ratio=%.1f%%\n",
               (unsigned long long)p, (unsigned long long)s,
               (p + s) > 0 ? 100.0 * p / (p + s) : 0.0);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_serving_adaptive_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
