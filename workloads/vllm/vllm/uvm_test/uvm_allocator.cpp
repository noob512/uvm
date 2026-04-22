/*
 * CUDA UVM Allocator for vLLM
 *
 * This is a custom CUDA allocator that uses cudaMallocManaged
 * to enable Unified Virtual Memory (UVM) in vLLM.
 *
 * UVM allows memory oversubscription - allocating more GPU memory
 * than physically available by using CPU memory as backing store.
 *
 * Usage:
 *   Set environment variable VLLM_USE_UVM=1 before starting vLLM
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

extern "C" {

// Global memory statistics (using C++ std::atomic)
static std::atomic<size_t> total_allocated{0};
static std::atomic<size_t> peak_allocated{0};
static std::atomic<size_t> num_allocs{0};
static std::atomic<size_t> num_frees{0};

// Configuration
static int enable_prefetch = 0;  // Whether to prefetch to device after allocation
static int verbose_logging = 0;  // Whether to log allocations
static size_t trace_min_bytes = 1 * 1024 * 1024;  // Trace allocations >= 1 MiB

// Log file handling
static FILE* log_file = nullptr;
static std::mutex log_mutex;
static std::chrono::steady_clock::time_point start_time;
static bool log_initialized = false;
static std::string current_phase = "unscoped";

struct AllocationInfo {
    size_t size;
    int device;
    size_t alloc_id;
    std::string phase;
    double alloc_elapsed_seconds;
};

static std::unordered_map<void*, AllocationInfo> active_allocations;

/**
 * Get current timestamp string
 */
static void get_timestamp(char* buffer, size_t size) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    struct tm* tm_info = localtime(&time_t_now);
    size_t len = strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
    snprintf(buffer + len, size - len, ".%03ld", (long)ms.count());
}

/**
 * Get elapsed time since start in seconds
 */
static double get_elapsed_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time).count();
}

static size_t read_trace_min_bytes_from_env() {
    const char* raw = getenv("VLLM_UVM_TRACE_MIN_BYTES");
    if (!raw || !*raw) {
        return trace_min_bytes;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 10);
    if (end_ptr == raw || parsed == 0) {
        return trace_min_bytes;
    }
    return static_cast<size_t>(parsed);
}

/**
 * Initialize log file (called once on first allocation)
 */
static void init_log_file() {
    if (log_initialized) return;

    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_initialized) return;  // Double-check after acquiring lock

    start_time = std::chrono::steady_clock::now();
    trace_min_bytes = read_trace_min_bytes_from_env();

    // Check environment variable for log file path
    const char* log_path = getenv("VLLM_UVM_LOG_FILE");
    if (!log_path) {
        log_path = "vllm_uvm_allocations.log";
    }

    log_file = fopen(log_path, "a");
    if (log_file) {
        char timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));
        fprintf(log_file, "\n========================================\n");
        fprintf(log_file, "[%s] vLLM UVM Allocator Session Started\n", timestamp);
        fprintf(log_file, "[%s] trace_min_bytes=%zu current_phase=%s\n",
                timestamp, trace_min_bytes, current_phase.c_str());
        fprintf(log_file, "========================================\n");
        fflush(log_file);
    } else {
        fprintf(stderr, "[vLLM UVM] Warning: Could not open log file: %s\n", log_path);
    }

    log_initialized = true;
}

/**
 * Log allocation to file
 */
static void log_allocation(const char* type, size_t size, size_t alloc_num,
                           size_t current_total, size_t peak, int device) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();

    fprintf(log_file, "[%s] [+%.3fs] %s #%zu: %.2f MB | device: %d | total: %.2f GB | peak: %.2f GB\n",
            timestamp, elapsed, type, alloc_num, size / 1e6, device,
            current_total / 1e9, peak / 1e9);
    fflush(log_file);
}

/**
 * Log free to file
 */
static void log_free(size_t size, size_t free_num, size_t current_total, int device) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();

    fprintf(log_file, "[%s] [+%.3fs] FREE #%zu: %.2f MB | device: %d | total: %.2f GB\n",
            timestamp, elapsed, free_num, size / 1e6, device, current_total / 1e9);
    fflush(log_file);
}

static void log_phase_event(const char* event, const char* phase) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    const char* safe_phase = (phase && phase[0] != '\0') ? phase : "unscoped";
    const char* safe_event = (event && event[0] != '\0') ? event : "marker";

    fprintf(log_file, "[%s] [+%.3fs] TRACE_PHASE event=%s phase=%s\n",
            timestamp, elapsed, safe_event, safe_phase);
    fflush(log_file);
}

static void trace_allocation_event(void* ptr,
                                   size_t size,
                                   size_t alloc_num,
                                   size_t current_total,
                                   size_t peak,
                                   int device,
                                   const std::string& phase) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = start + size - 1;

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_ALLOC alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_mb=%.6f device=%d phase=%s total_bytes=%zu peak_bytes=%zu\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size / 1e6,
        device,
        phase.c_str(),
        current_total,
        peak
    );
    fflush(log_file);
}

static void trace_free_event(void* ptr,
                             size_t size,
                             size_t free_num,
                             size_t current_total,
                             int device,
                             const AllocationInfo* info,
                             const std::string& current_phase_snapshot) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = start + size - 1;
    double lifetime_seconds =
        info ? (elapsed - info->alloc_elapsed_seconds) : -1.0;

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_FREE free_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_mb=%.6f device=%d phase=%s alloc_id=%zu "
        "alloc_phase=%s lifetime_s=%.6f total_bytes=%zu\n",
        timestamp,
        elapsed,
        free_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size / 1e6,
        device,
        current_phase_snapshot.c_str(),
        info ? info->alloc_id : 0,
        info ? info->phase.c_str() : "unknown",
        lifetime_seconds,
        current_total
    );
    fflush(log_file);
}

/**
 * Allocate CUDA managed (UVM) memory
 *
 * This function is called by PyTorch's pluggable allocator interface.
 *
 * @param size Size in bytes to allocate
 * @param device CUDA device ID
 * @param stream CUDA stream (unused for managed memory)
 * @return Pointer to allocated memory, or NULL on failure
 */
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    // Initialize log file on first allocation
    if (!log_initialized) {
        init_log_file();
    }

    void* ptr = NULL;
    cudaError_t err;

    // Use cudaMallocManaged for UVM
    // cudaMemAttachGlobal makes memory accessible from any GPU and CPU
    err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMallocManaged failed for %zd bytes: %s\n",
                size, cudaGetErrorString(err));
        return NULL;
    }

    // Update statistics
    size_t current = total_allocated.fetch_add(size) + size;
    size_t alloc_count = num_allocs.fetch_add(1) + 1;

    // Update peak if needed (lock-free)
    size_t peak = peak_allocated.load();
    while (current > peak) {
        if (peak_allocated.compare_exchange_weak(peak, current)) {
            break;
        }
    }

    std::string phase_snapshot;
    double alloc_elapsed = get_elapsed_seconds();
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;
        if (size >= trace_min_bytes) {
            active_allocations[ptr] = AllocationInfo{
                static_cast<size_t>(size),
                device,
                alloc_count,
                phase_snapshot,
                alloc_elapsed,
            };
        }
    }

    // Log large allocations (> 100MB) to file
    if (size > 100 * 1024 * 1024) {
        log_allocation("ALLOC", size, alloc_count, current, peak_allocated.load(), device);
    }

    if (size >= trace_min_bytes) {
        trace_allocation_event(
            ptr,
            static_cast<size_t>(size),
            alloc_count,
            current,
            peak_allocated.load(),
            device,
            phase_snapshot
        );
    }

    // Also log to stderr if verbose logging is enabled
    if (verbose_logging && size > 100 * 1024 * 1024) {
        fprintf(stderr, "[vLLM UVM] Alloc #%zu: %.2f MB (total: %.2f GB, peak: %.2f GB)\n",
                alloc_count, size / 1e6, current / 1e9,
                peak_allocated.load() / 1e9);
    }

    // Optionally prefetch to device
    // This can improve performance by proactively moving data to GPU
    if (enable_prefetch && device >= 0 && ptr != NULL) {
        cudaMemPrefetchAsync(ptr, size, device, stream);
    }

    return ptr;
}

/**
 * Free CUDA managed memory
 *
 * @param ptr Pointer to memory to free
 * @param size Size of allocation (for statistics)
 * @param device CUDA device ID (unused)
 * @param stream CUDA stream (unused)
 */
void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    if (ptr != NULL) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vLLM UVM] cudaFree failed: %s\n",
                    cudaGetErrorString(err));
        }

        // Update statistics
        size_t current = total_allocated.fetch_sub(size) - size;
        size_t free_count = num_frees.fetch_add(1) + 1;
        AllocationInfo info{};
        bool has_info = false;
        std::string phase_snapshot;

        {
            std::lock_guard<std::mutex> lock(log_mutex);
            phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;
            auto it = active_allocations.find(ptr);
            if (it != active_allocations.end()) {
                info = it->second;
                has_info = true;
                active_allocations.erase(it);
            }
        }

        // Log large frees (> 100MB) to file
        if (size > 100 * 1024 * 1024) {
            log_free(size, free_count, current, device);
        }

        if (static_cast<size_t>(size) >= trace_min_bytes || has_info) {
            trace_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                current,
                device,
                has_info ? &info : nullptr,
                phase_snapshot
            );
        }
    }
}

void uvm_set_phase(const char* phase) {
    if (!log_initialized) {
        init_log_file();
    }

    const char* safe_phase = (phase && phase[0] != '\0') ? phase : "unscoped";
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        current_phase = safe_phase;
    }
    log_phase_event("set", safe_phase);
}

void uvm_mark_phase_event(const char* event, const char* phase) {
    if (!log_initialized) {
        init_log_file();
    }

    if (phase && phase[0] != '\0') {
        log_phase_event(event, phase);
        return;
    }

    std::string phase_snapshot;
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;
    }
    log_phase_event(event, phase_snapshot.c_str());
}

// =============================================================================
// Statistics API (can be called from Python via ctypes)
// =============================================================================

/**
 * Get current allocated bytes
 */
size_t uvm_get_allocated_bytes(void) {
    return total_allocated.load();
}

/**
 * Get peak allocated bytes
 */
size_t uvm_get_peak_allocated_bytes(void) {
    return peak_allocated.load();
}

/**
 * Get total number of allocations
 */
size_t uvm_get_num_allocs(void) {
    return num_allocs.load();
}

/**
 * Get total number of frees
 */
size_t uvm_get_num_frees(void) {
    return num_frees.load();
}

/**
 * Reset peak statistics to current allocation
 */
void uvm_reset_peak_stats(void) {
    peak_allocated.store(total_allocated.load());
}

/**
 * Reset all statistics
 */
void uvm_reset_all_stats(void) {
    total_allocated.store(0);
    peak_allocated.store(0);
    num_allocs.store(0);
    num_frees.store(0);
}

/**
 * Enable/disable prefetching
 */
void uvm_set_prefetch(int enabled) {
    enable_prefetch = enabled;
}

/**
 * Enable/disable verbose logging
 */
void uvm_set_verbose(int enabled) {
    verbose_logging = enabled;
}

/**
 * Flush and close the log file, writing a summary
 */
void uvm_close_log(void) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();

    fprintf(log_file, "========================================\n");
    fprintf(log_file, "[%s] Session Summary (duration: %.2fs)\n", timestamp, elapsed);
    fprintf(log_file, "  Total allocations: %zu\n", num_allocs.load());
    fprintf(log_file, "  Total frees: %zu\n", num_frees.load());
    fprintf(log_file, "  Current allocated: %.2f GB\n", total_allocated.load() / 1e9);
    fprintf(log_file, "  Peak allocated: %.2f GB\n", peak_allocated.load() / 1e9);
    fprintf(log_file, "========================================\n\n");

    fflush(log_file);
    fclose(log_file);
    log_file = nullptr;
    log_initialized = false;
}

/**
 * Set custom log file path (must be called before first allocation)
 */
void uvm_set_log_file(const char* path) {
    if (log_initialized) {
        fprintf(stderr, "[vLLM UVM] Warning: Cannot change log file after initialization\n");
        return;
    }
    // The log path is read from VLLM_UVM_LOG_FILE environment variable in init_log_file()
    // This function is provided for programmatic control if needed
    if (path) {
        setenv("VLLM_UVM_LOG_FILE", path, 1);
    }
}

/**
 * Prefetch memory region to a specific device
 *
 * @param ptr Pointer to memory region
 * @param size Size of region in bytes
 * @param device Target device (-1 for CPU, >= 0 for GPU)
 * @param stream CUDA stream for async prefetch
 */
void uvm_prefetch(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr != NULL && size > 0) {
        cudaMemPrefetchAsync(ptr, size, device, stream);
    }
}

/**
 * Set memory advice for a region
 *
 * @param ptr Pointer to memory region
 * @param size Size of region in bytes
 * @param advice One of: 1=ReadMostly, 2=PreferredLocation, 3=AccessedBy
 * @param device Device ID for the advice
 */
void uvm_advise(void* ptr, size_t size, int advice, int device) {
    if (ptr == NULL || size == 0) return;

    cudaMemoryAdvise cuda_advice;
    switch (advice) {
        case 1:
            cuda_advice = cudaMemAdviseSetReadMostly;
            break;
        case 2:
            cuda_advice = cudaMemAdviseSetPreferredLocation;
            break;
        case 3:
            cuda_advice = cudaMemAdviseSetAccessedBy;
            break;
        default:
            fprintf(stderr, "[vLLM UVM] Unknown advice type: %d\n", advice);
            return;
    }

    cudaError_t err = cudaMemAdvise(ptr, size, cuda_advice, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMemAdvise failed: %s\n",
                cudaGetErrorString(err));
    }
}

}  // extern "C"
