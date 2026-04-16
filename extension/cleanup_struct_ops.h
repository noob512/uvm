/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _CLEANUP_STRUCT_OPS_H
#define _CLEANUP_STRUCT_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stddef.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#define UVM_STRUCT_OPS_NAME_PREFIX "uvm_ops"
#define UVM_STRUCT_OPS_MAX_TRACKED 16

struct uvm_struct_ops_instance {
    __u32 id;
    char name[BPF_OBJ_NAME_LEN];
};

static inline int collect_uvm_struct_ops_instances(struct uvm_struct_ops_instance *instances,
                                                   size_t max_instances)
{
    __u32 map_id = 0;
    int count = 0;

    while (1) {
        struct bpf_map_info info = {};
        __u32 len = sizeof(info);
        int err;
        int fd;

        err = bpf_map_get_next_id(map_id, &map_id);
        if (err) {
            if (errno == ENOENT)
                break;

            fprintf(stderr, "Skipping struct_ops state scan: %s\n", strerror(errno));
            break;
        }

        fd = bpf_map_get_fd_by_id(map_id);
        if (fd < 0)
            continue;

        err = bpf_obj_get_info_by_fd(fd, &info, &len);
        if (err) {
            close(fd);
            continue;
        }

        if (info.type == BPF_MAP_TYPE_STRUCT_OPS &&
            strncmp(info.name, UVM_STRUCT_OPS_NAME_PREFIX,
                    strlen(UVM_STRUCT_OPS_NAME_PREFIX)) == 0) {
            if (instances && (size_t)count < max_instances) {
                instances[count].id = info.id;
                snprintf(instances[count].name, sizeof(instances[count].name),
                         "%s", info.name);
            }
            count++;
        }

        close(fd);
    }

    return count;
}

static inline void print_uvm_struct_ops_instances(const char *label,
                                                  const struct uvm_struct_ops_instance *instances,
                                                  int count)
{
    int i;

    if (count <= 0) {
        printf("[%s] No UVM struct_ops instances found.\n", label);
        return;
    }

    fprintf(stderr, "[%s] Found %d UVM struct_ops instance(s):\n", label, count);
    for (i = 0; i < count && i < UVM_STRUCT_OPS_MAX_TRACKED; ++i) {
        fprintf(stderr, "  - id=%u name=%s\n", instances[i].id, instances[i].name);
    }
    if (count > UVM_STRUCT_OPS_MAX_TRACKED) {
        fprintf(stderr, "  ... plus %d more instance(s) not shown\n",
                count - UVM_STRUCT_OPS_MAX_TRACKED);
    }
}

static inline int verify_no_uvm_struct_ops_instances(const char *label)
{
    struct uvm_struct_ops_instance instances[UVM_STRUCT_OPS_MAX_TRACKED];
    int count;

    memset(instances, 0, sizeof(instances));
    count = collect_uvm_struct_ops_instances(instances, UVM_STRUCT_OPS_MAX_TRACKED);
    if (count > 0) {
        print_uvm_struct_ops_instances(label, instances, count);
        return -EEXIST;
    }

    printf("[%s] Verified: no UVM struct_ops instances remain.\n", label);
    return 0;
}

/* Cleanup old struct_ops instances
 * Returns: 0 on success (no old instances found)
 *          -EEXIST if old instances were found
 */
static inline int cleanup_old_struct_ops(void) {
    struct uvm_struct_ops_instance instances[UVM_STRUCT_OPS_MAX_TRACKED];
    __u32 map_id = 0;
    int cleaned = 0;
    int count = 0;
    int err;

    printf("Checking for old struct_ops instances...\n");
    memset(instances, 0, sizeof(instances));

    /* Iterate through all BPF maps */
    while (1) {
        struct bpf_map_info info = {};
        __u32 len = sizeof(info);
        int fd;

        err = bpf_map_get_next_id(map_id, &map_id);
        if (err) {
            if (errno == ENOENT) {
                break; /* No more maps */
            }
            fprintf(stderr, "Skipping struct_ops cleanup scan: %s\n", strerror(errno));
            break;
        }

        fd = bpf_map_get_fd_by_id(map_id);
        if (fd < 0) {
            continue;
        }

        err = bpf_obj_get_info_by_fd(fd, &info, &len);
        if (err) {
            close(fd);
            continue;
        }

        /* Check if this is any UVM struct_ops map (covers all gpu_ext policies) */
        if (info.type == BPF_MAP_TYPE_STRUCT_OPS &&
            strncmp(info.name, UVM_STRUCT_OPS_NAME_PREFIX,
                    strlen(UVM_STRUCT_OPS_NAME_PREFIX)) == 0) {
            printf("Found old struct_ops map (ID: %u, name: %s)\n",
                   info.id, info.name);
            printf("Attempting to clean up...\n");

            if (count < UVM_STRUCT_OPS_MAX_TRACKED) {
                instances[count].id = info.id;
                snprintf(instances[count].name, sizeof(instances[count].name),
                         "%s", info.name);
            }
            count++;

            /* Try to pin and then unpin to release references */
            char pin_path[256];
            snprintf(pin_path, sizeof(pin_path),
                     "/sys/fs/bpf/old_testmod_%u", info.id);

            /* Pin the map */
            if (bpf_obj_pin(fd, pin_path) == 0) {
                /* Immediately unpin it */
                unlink(pin_path);
                printf("Cleaned up pinned reference\n");
            }

            cleaned++;
        }

        close(fd);
    }

    if (cleaned > 0) {
        printf("Found %d old struct_ops instance(s)\n", cleaned);
        printf("Note: Old instances may still be active if held by running processes.\n");
        printf("Please kill any running struct_ops processes first.\n");
        print_uvm_struct_ops_instances("cleanup_old_struct_ops", instances, count);
        return -EEXIST;
    }

    printf("No old struct_ops instances found.\n");
    return 0;
}

#endif /* _CLEANUP_STRUCT_OPS_H */
