/*******************************************************************************
    Copyright (c) 2016 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_test.h"
#include "uvm_perf_events.h"
#include "uvm_va_space.h"

// Entry of the event callback list
typedef struct
{
    uvm_perf_event_callback_t callback;

    struct list_head callback_list_node;
} callback_desc_t;

// Cache for callback descriptor list entries
static struct kmem_cache *g_callback_desc_cache;

// Check if the callback list already contains an entry for the given callback. Caller needs to hold (at least) read
// va_space_events lock
static callback_desc_t *event_list_find_callback(uvm_perf_va_space_events_t *va_space_events,
                                                 struct list_head *callback_list, uvm_perf_event_callback_t callback)
{
    callback_desc_t *callback_desc;

    uvm_assert_rwsem_locked(&va_space_events->lock);

    list_for_each_entry(callback_desc, callback_list, callback_list_node) {
        if (callback_desc->callback == callback)
            return callback_desc;
    }

    return NULL;
}

NV_STATUS uvm_perf_register_event_callback_locked(uvm_perf_va_space_events_t *va_space_events,
                                                  uvm_perf_event_t event_id,
                                                  uvm_perf_event_callback_t callback)
{
    callback_desc_t *callback_desc;
    struct list_head *callback_list;

    UVM_ASSERT(event_id >= 0 && event_id < UVM_PERF_EVENT_COUNT);
    UVM_ASSERT(callback);

    uvm_assert_rwsem_locked_write(&va_space_events->lock);

    callback_list = &va_space_events->event_callbacks[event_id];

    UVM_ASSERT(!event_list_find_callback(va_space_events, callback_list, callback));

    callback_desc = kmem_cache_alloc(g_callback_desc_cache, NV_UVM_GFP_FLAGS);
    if (!callback_desc)
        return NV_ERR_NO_MEMORY;

    callback_desc->callback = callback;
    list_add_tail(&callback_desc->callback_list_node, callback_list);

    return NV_OK;
}

NV_STATUS uvm_perf_register_event_callback(uvm_perf_va_space_events_t *va_space_events, uvm_perf_event_t event_id,
                                           uvm_perf_event_callback_t callback)
{
    NV_STATUS status;

    uvm_down_write(&va_space_events->lock);
    status = uvm_perf_register_event_callback_locked(va_space_events, event_id, callback);
    uvm_up_write(&va_space_events->lock);

    return status;
}

void uvm_perf_unregister_event_callback_locked(uvm_perf_va_space_events_t *va_space_events, uvm_perf_event_t event_id,
                                               uvm_perf_event_callback_t callback)
{
    callback_desc_t *callback_desc;
    struct list_head *callback_list;

    UVM_ASSERT(event_id >= 0 && event_id < UVM_PERF_EVENT_COUNT);
    UVM_ASSERT(callback);

    uvm_assert_rwsem_locked_write(&va_space_events->lock);

    callback_list = &va_space_events->event_callbacks[event_id];
    callback_desc = event_list_find_callback(va_space_events, callback_list, callback);

    if (!callback_desc)
        return;

    list_del(&callback_desc->callback_list_node);

    kmem_cache_free(g_callback_desc_cache, callback_desc);
}

void uvm_perf_unregister_event_callback(uvm_perf_va_space_events_t *va_space_events, uvm_perf_event_t event_id,
                                        uvm_perf_event_callback_t callback)
{
    uvm_down_write(&va_space_events->lock);
    uvm_perf_unregister_event_callback_locked(va_space_events, event_id, callback);
    uvm_up_write(&va_space_events->lock);
}

/**
 * 广播 UVM 性能事件给所有已注册的回调函数（订阅者）。
 *
 * @param va_space_events 指向当前虚拟地址空间（VA Space）性能事件管理器的指针。
 * 它维护了所有的回调函数列表和并发锁。
 * @param event_id        当前触发的事件类型 ID（例如：UVM_PERF_EVENT_FAULT 表示缺页，
 * UVM_PERF_EVENT_MIGRATION 表示内存迁移）。
 * @param event_data      事件的详细数据载荷（Payload），包含了前一个函数打包好的 
 * GPU ID、发生地址、重复次数等详尽上下文。
 */
void uvm_perf_event_notify(uvm_perf_va_space_events_t *va_space_events, uvm_perf_event_t event_id,
                           uvm_perf_event_data_t *event_data)
{
    callback_desc_t *callback_desc;
    struct list_head *callback_list;

    // ---------------------------------------------------------
    // 第一步：参数合法性校验（防呆设计）
    // ---------------------------------------------------------
    // 确保 event_id 在合法范围内，且事件数据指针不为空。
    // UVM_ASSERT 在 Debug 构建中会触发内核 BUG，在 Release 中通常会被优化掉以保证性能。
    UVM_ASSERT(event_id >= 0 && event_id < UVM_PERF_EVENT_COUNT);
    UVM_ASSERT(event_data);

    // ---------------------------------------------------------
    // 第二步：定位订阅者列表
    // ---------------------------------------------------------
    // event_callbacks 是一个数组，每个元素对应一种事件类型的链表头。
    // 这里以 O(1) 的时间复杂度直接取出对当前 event_id 感兴趣的订阅者链表。
    callback_list = &va_space_events->event_callbacks[event_id];

    // ---------------------------------------------------------
    // 第三步：获取读写锁（Read Lock）
    // ---------------------------------------------------------
    // uvm_down_read 申请获取读写信号量（rw_semaphore）的“读锁”。
    // 为什么用读锁？
    // 因为触发事件的频率极高（可能多个 GPU 线程同时在触发），允许多个线程
    // 同时读取（遍历）回调列表，不会发生互相阻塞，极大提升了并发性能。
    // 只有当有新的工具想要“注册”或“注销”监听时，才需要申请“写锁”（独占锁）。
    uvm_down_read(&va_space_events->lock);

    // ---------------------------------------------------------
    // 第四步：遍历并执行回调（广播过程）
    // ---------------------------------------------------------
    // list_for_each_entry 是 Linux 内核中用于遍历双向链表的经典宏。
    // 它遍历 callback_list 中的每一个订阅者节点 (callback_desc)。
    list_for_each_entry(callback_desc, callback_list, callback_list_node) {
        // 调用订阅者注册的函数指针 (callback)，将事件类型和数据作为参数传递给它。
        // 这就是实际将数据送达（比如送给 Nsight Systems 驱动层）的动作。
        callback_desc->callback(event_id, event_data);
    }

    // ---------------------------------------------------------
    // 第五步：释放锁
    // ---------------------------------------------------------
    // 释放读锁，结束本次广播。
    uvm_up_read(&va_space_events->lock);
}

bool uvm_perf_is_event_callback_registered(uvm_perf_va_space_events_t *va_space_events,
                                           uvm_perf_event_t event_id,
                                           uvm_perf_event_callback_t callback)
{
    callback_desc_t *callback_desc;
    struct list_head *callback_list;

    uvm_assert_rwsem_locked(&va_space_events->lock);

    callback_list = &va_space_events->event_callbacks[event_id];
    callback_desc = event_list_find_callback(va_space_events, callback_list, callback);

    return callback_desc != NULL;
}

NV_STATUS uvm_perf_init_va_space_events(uvm_va_space_t *va_space, uvm_perf_va_space_events_t *va_space_events)
{
    unsigned event_id;

    uvm_init_rwsem(&va_space_events->lock, UVM_LOCK_ORDER_VA_SPACE_EVENTS);

    // Initialize event callback lists
    for (event_id = 0; event_id < UVM_PERF_EVENT_COUNT; ++event_id)
        INIT_LIST_HEAD(&va_space_events->event_callbacks[event_id]);

    va_space_events->va_space = va_space;

    return NV_OK;
}

void uvm_perf_destroy_va_space_events(uvm_perf_va_space_events_t *va_space_events)
{
    unsigned event_id;

    // If the va_space member was not set, va_space creation failed before initializing its va_space_events member. We
    // are done.
    if (!va_space_events->va_space)
        return;

    // Destroy all event callback lists' entries
    for (event_id = 0; event_id < UVM_PERF_EVENT_COUNT; ++event_id) {
        callback_desc_t *callback_desc, *callback_desc_tmp;
        struct list_head *callback_list;

        callback_list = &va_space_events->event_callbacks[event_id];

        list_for_each_entry_safe(callback_desc, callback_desc_tmp, callback_list, callback_list_node) {
            list_del(&callback_desc->callback_list_node);
            kmem_cache_free(g_callback_desc_cache, callback_desc);
        }
    }

    va_space_events->va_space = NULL;
}

NV_STATUS uvm_perf_events_init(void)
{
    g_callback_desc_cache = NV_KMEM_CACHE_CREATE("uvm_perf_callback_list", callback_desc_t);
    if (!g_callback_desc_cache)
        return NV_ERR_NO_MEMORY;

    return NV_OK;
}

void uvm_perf_events_exit(void)
{
    kmem_cache_destroy_safe(&g_callback_desc_cache);
}
