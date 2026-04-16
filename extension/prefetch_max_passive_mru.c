/* Combined Always-Max Prefetch + Passive MRU Eviction - Loader */
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "prefetch_max_passive_mru.skel.h"
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
    struct prefetch_max_passive_mru_bpf *skel;
    struct bpf_link *link;
    int err;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    libbpf_set_print(libbpf_print_fn);
    cleanup_old_struct_ops();

    skel = prefetch_max_passive_mru_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = prefetch_max_passive_mru_bpf__load(skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_max_passive_mru);
    if (!link) {
        err = -errno;
        fprintf(stderr, "Failed to attach struct_ops: %s (%d)\n", strerror(-err), err);
        goto cleanup;
    }

    printf("Loaded: always_max prefetch + passive MRU eviction\n");
    printf("  T1 threshold: 3 accesses → protect (move_tail)\n");
    printf("  Non-T1: BYPASS without move (passive MRU)\n");
    printf("Press Ctrl-C to exit...\n");

    while (!exiting) {
        sleep(1);
    }

    printf("\nDetaching struct_ops...\n");
    bpf_link__destroy(link);

cleanup:
    prefetch_max_passive_mru_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
