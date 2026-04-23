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
#include <sys/stat.h>
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
static int policy_enabled = 1;  // Trace-only policy classification is enabled by default
static int policy_warmup_prefetch_enabled = 0;
static int policy_warmup_advise_gpu = 0;
static size_t policy_warmup_prefetch_min_bytes = 1 * 1024 * 1024;
static std::string policy_mode = "trace_only";
static int unknown_detail_enabled = 0;
static size_t unknown_detail_min_bytes = 0;
static int gap_watch_enabled = 0;
static uintptr_t gap_watch_start = 0;
static uintptr_t gap_watch_end = 0;
static std::string gap_watch_name = "unnamed_gap";
static int gap_watch_all_classes = 1;
static size_t gap_watch_min_bytes = 0;
static std::string gap_watch_control_file = "";
static size_t gap_watch_refresh_ms = 250;
static bool gap_watch_control_seen = false;
static uint64_t gap_watch_control_mtime_ns = 0;
static off_t gap_watch_control_size = -1;
static std::chrono::steady_clock::time_point gap_watch_last_refresh_check =
    std::chrono::steady_clock::time_point::min();

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
    std::string alloc_class_name;
    std::string policy_action_name;
    std::string size_bucket;
    bool unknown_detail_logged;
    bool gap_watch_logged;
    uintptr_t gap_overlap_start;
    uintptr_t gap_overlap_end;
    size_t gap_overlap_bytes;
};

static std::unordered_map<void*, AllocationInfo> active_allocations;

enum class AllocationClass {
    /**
     * 1. 模型权重 (Model Weights)
     * 特点：只读、生命周期极长（随模型加载而存在，随程序结束而释放）。
     * 策略：由于每一轮推理（Forward Pass）都要读取全部权重，这类内存绝对不能被换出到 CPU，
     * 否则会因为 PCIe 带宽瓶颈导致吞吐量断崖式下跌。
     */
    WeightPersistent,

    /**
     * 2. KV 缓存 (Key-Value Cache)
     * 特点：读写频繁、空间占用最大、生命周期长。
     * 策略：这是 vLLM 的核心显存占用者。
     */
    KvPersistent,

    /**
     * 3. 预热期工作区 (Warmup/Profile Workspace)
     * 特点：在模型启动阶段（Warmup）大量产生，用于算子自动调优 (Autotune) 或初始状态设置。
     */
    WarmupWorkspace,

    /**
     * 4. 运行时临时碎片 (Runtime Scratch)
     * 特点：极小（通常 < 1MB）、存活时间极短（可能一个算子执行完就释放）。
     * 策略：这类内存分配非常频繁。由于其尺寸小且释放快，通常不需要进行 UVM 预取，
     */
    RuntimeScratch,

    /**
     * 5. 运行时计算空间 (Runtime Workspace)
     * 特点：中等或大尺寸（如中间激活值 Activations、Logits）、存活于单次推理迭代中。
     * 策略：这些是计算时的“草稿纸”。它们在推理时需要极高的读写速度，但在两次推理请求之间
     */
    RuntimeWorkspace,

    /**
     * 6. 未知托管内存 (Unknown Managed)
     * 特点：未能命中任何启发式规则的分配请求。
     * 策略：作为兜底类别。如果日志中大量出现此类别，说明现有的 classify_allocation 
     * 匹配规则（如 Phase 名称、Size 阈值）需要更新。
     */
    UnknownManaged,
};

enum class PolicyAction {
    ManagedDefault,//默认动作
    ManagedPrefetchGpu,//为阶段 1 预留,阶段 0 默认不会触发
};

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

static bool read_bool_from_env(const char* name, bool default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    return strcmp(raw, "1") == 0 || strcasecmp(raw, "true") == 0 ||
           strcasecmp(raw, "yes") == 0 || strcasecmp(raw, "on") == 0;
}

static size_t read_size_from_env(const char* name, size_t default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 10);
    if (end_ptr == raw || parsed == 0) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

static bool read_hex_u64_from_env(const char* name, uintptr_t* value_out) {
    const char* raw = getenv(name);
    if (!raw || !*raw || !value_out) {
        return false;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 0);
    if (end_ptr == raw || *end_ptr != '\0') {
        return false;
    }

    *value_out = static_cast<uintptr_t>(parsed);
    return true;
}

static uint64_t stat_mtime_ns(const struct stat& st) {
    return static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(st.st_mtim.tv_nsec);
}

static std::string read_string_from_env(const char* name,
                                        const char* default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    return raw;
}

static bool phase_contains(const std::string& phase, const char* needle) {
    return phase.find(needle) != std::string::npos;
}

static std::string trim_copy(const std::string& input) {
    size_t start = 0;
    while (start < input.size() &&
           (input[start] == ' ' || input[start] == '\t' ||
            input[start] == '\r' || input[start] == '\n')) {
        ++start;
    }

    size_t end = input.size();
    while (end > start &&
           (input[end - 1] == ' ' || input[end - 1] == '\t' ||
            input[end - 1] == '\r' || input[end - 1] == '\n')) {
        --end;
    }

    return input.substr(start, end - start);
}

static bool parse_bool_string(const std::string& value, bool* out) {
    if (!out) {
        return false;
    }
    if (value == "1" || value == "true" || value == "True" ||
        value == "TRUE" || value == "yes" || value == "on") {
        *out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "False" ||
        value == "FALSE" || value == "no" || value == "off") {
        *out = false;
        return true;
    }
    return false;
}

static bool parse_size_string(const std::string& value, size_t* out) {
    if (!out) {
        return false;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &end_ptr, 10);
    if (end_ptr == value.c_str() || *end_ptr != '\0') {
        return false;
    }

    *out = static_cast<size_t>(parsed);
    return true;
}

static bool parse_hex_string(const std::string& value, uintptr_t* out) {
    if (!out) {
        return false;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &end_ptr, 0);
    if (end_ptr == value.c_str() || *end_ptr != '\0') {
        return false;
    }

    *out = static_cast<uintptr_t>(parsed);
    return true;
}

static const char* allocation_class_to_string(AllocationClass alloc_class) {
    switch (alloc_class) {
        case AllocationClass::WeightPersistent:
            return "weight_persistent";
        case AllocationClass::KvPersistent:
            return "kv_persistent";
        case AllocationClass::WarmupWorkspace:
            return "warmup_workspace";
        case AllocationClass::RuntimeScratch:
            return "runtime_scratch";
        case AllocationClass::RuntimeWorkspace:
            return "runtime_workspace";
        case AllocationClass::UnknownManaged:
        default:
            return "unknown_managed";
    }
}

static const char* policy_action_to_string(PolicyAction action) {
    switch (action) {
        case PolicyAction::ManagedPrefetchGpu:
            return "managed_prefetch_gpu";
        case PolicyAction::ManagedDefault:
        default:
            return "managed_default";
    }
}

static const char* size_bucket_for(size_t size) {
    if (size < 64 * 1024) return "<64KiB";
    if (size < 256 * 1024) return "64KiB-256KiB";
    if (size < 1 * 1024 * 1024) return "256KiB-1MiB";
    if (size < 2 * 1024 * 1024) return "1MiB-2MiB";
    if (size < 4 * 1024 * 1024) return "2MiB-4MiB";
    if (size < 8 * 1024 * 1024) return "4MiB-8MiB";
    if (size < 16 * 1024 * 1024) return "8MiB-16MiB";
    return ">=16MiB";
}

static uintptr_t region_end_for(uintptr_t start, size_t size) {
    if (size == 0) {
        return start;
    }
    return start + size - 1;
}

struct GapWatchConfigSnapshot {
    bool enabled;
    uintptr_t start;
    uintptr_t end;
    std::string name;
    bool all_classes;
    size_t min_bytes;
};

static GapWatchConfigSnapshot current_gap_watch_config_snapshot() {
    return GapWatchConfigSnapshot{
        gap_watch_enabled != 0,
        gap_watch_start,
        gap_watch_end,
        gap_watch_name,
        gap_watch_all_classes != 0,
        gap_watch_min_bytes,
    };
}

static bool gap_watch_configs_equal(const GapWatchConfigSnapshot& left,
                                    const GapWatchConfigSnapshot& right) {
    return left.enabled == right.enabled &&
           left.start == right.start &&
           left.end == right.end &&
           left.name == right.name &&
           left.all_classes == right.all_classes &&
           left.min_bytes == right.min_bytes;
}

static void trace_gap_watch_config_event_locked(const char* source,
                                                const char* event,
                                                const GapWatchConfigSnapshot& config,
                                                const char* reason) {
    if (!log_file) {
        return;
    }

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    const char* safe_source =
        (source && source[0] != '\0') ? source : "unknown_source";
    const char* safe_event =
        (event && event[0] != '\0') ? event : "applied";
    const char* safe_reason =
        (reason && reason[0] != '\0') ? reason : "none";

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_CONFIG source=%s event=%s "
        "enabled=%d name=%s start=0x%llx end=0x%llx all_classes=%d "
        "min_bytes=%zu reason=%s\n",
        timestamp,
        elapsed,
        safe_source,
        safe_event,
        config.enabled ? 1 : 0,
        config.name.c_str(),
        static_cast<unsigned long long>(config.start),
        static_cast<unsigned long long>(config.end),
        config.all_classes ? 1 : 0,
        config.min_bytes,
        safe_reason
    );
    fflush(log_file);
}

static void apply_gap_watch_config_locked(const GapWatchConfigSnapshot& config,
                                          const char* source,
                                          const char* reason) {
    GapWatchConfigSnapshot previous = current_gap_watch_config_snapshot();
    if (gap_watch_configs_equal(previous, config)) {
        return;
    }

    gap_watch_enabled = config.enabled ? 1 : 0;
    gap_watch_start = config.start;
    gap_watch_end = config.end;
    gap_watch_name = config.name.empty() ? "unnamed_gap" : config.name;
    gap_watch_all_classes = config.all_classes ? 1 : 0;
    gap_watch_min_bytes = config.min_bytes;

    GapWatchConfigSnapshot updated = current_gap_watch_config_snapshot();
    trace_gap_watch_config_event_locked(source, "applied", updated, reason);
}

static bool compute_overlap(uintptr_t start,
                            size_t size,
                            uintptr_t other_start,
                            uintptr_t other_end,
                            uintptr_t* overlap_start_out,
                            uintptr_t* overlap_end_out,
                            size_t* overlap_bytes_out) {
    if (size == 0 || other_end < other_start) {
        return false;
    }

    uintptr_t end = region_end_for(start, size);
    uintptr_t overlap_start = start > other_start ? start : other_start;
    uintptr_t overlap_end = end < other_end ? end : other_end;
    if (overlap_start > overlap_end) {
        return false;
    }

    if (overlap_start_out) *overlap_start_out = overlap_start;
    if (overlap_end_out) *overlap_end_out = overlap_end;
    if (overlap_bytes_out) *overlap_bytes_out = overlap_end - overlap_start + 1;
    return true;
}

static bool should_trace_unknown_detail(AllocationClass alloc_class,
                                        size_t size) {
    if (!unknown_detail_enabled) {
        return false;
    }
    if (alloc_class != AllocationClass::UnknownManaged) {
        return false;
    }
    return size >= unknown_detail_min_bytes;
}

static bool should_trace_gap_watch(AllocationClass alloc_class,
                                   size_t size,
                                   size_t overlap_bytes) {
    if (!gap_watch_enabled || overlap_bytes == 0) {
        return false;
    }
    if (size < gap_watch_min_bytes) {
        return false;
    }
    return gap_watch_all_classes ||
           alloc_class == AllocationClass::UnknownManaged;
}

static void refresh_gap_watch_from_control_file_if_needed(bool force) {
    if (gap_watch_control_file.empty()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    if (!force &&
        gap_watch_last_refresh_check != std::chrono::steady_clock::time_point::min()) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - gap_watch_last_refresh_check
        ).count();
        if (elapsed_ms >= 0 &&
            static_cast<size_t>(elapsed_ms) < gap_watch_refresh_ms) {
            return;
        }
    }
    gap_watch_last_refresh_check = now;

    struct stat st {};
    if (stat(gap_watch_control_file.c_str(), &st) != 0) {
        return;
    }

    uint64_t mtime_ns = stat_mtime_ns(st);
    if (!force && gap_watch_control_seen &&
        mtime_ns == gap_watch_control_mtime_ns &&
        st.st_size == gap_watch_control_size) {
        return;
    }

    FILE* control = fopen(gap_watch_control_file.c_str(), "r");
    if (!control) {
        return;
    }

    GapWatchConfigSnapshot next = current_gap_watch_config_snapshot();
    next.enabled = false;
    next.start = 0;
    next.end = 0;

    bool saw_enabled = false;
    bool parsed_enabled = false;
    bool parsed_all_classes = false;
    bool parsed_name = false;
    bool parsed_start = false;
    bool parsed_end = false;
    bool parsed_min_bytes = false;
    bool enabled_value = false;
    bool all_classes_value = gap_watch_all_classes != 0;
    uintptr_t start_value = 0;
    uintptr_t end_value = 0;
    size_t min_bytes_value = gap_watch_min_bytes;
    std::string name_value = gap_watch_name;
    char buffer[1024];

    while (fgets(buffer, sizeof(buffer), control) != nullptr) {
        std::string line = trim_copy(buffer);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        std::string key = trim_copy(line.substr(0, eq_pos));
        std::string value = trim_copy(line.substr(eq_pos + 1));
        if (key == "enabled") {
            saw_enabled = true;
            parsed_enabled = parse_bool_string(value, &enabled_value);
        } else if (key == "all_classes") {
            parsed_all_classes = parse_bool_string(value, &all_classes_value);
        } else if (key == "name") {
            name_value = value;
            parsed_name = true;
        } else if (key == "start") {
            parsed_start = parse_hex_string(value, &start_value);
        } else if (key == "end") {
            parsed_end = parse_hex_string(value, &end_value);
        } else if (key == "min_bytes") {
            parsed_min_bytes = parse_size_string(value, &min_bytes_value);
        }
    }
    fclose(control);

    gap_watch_control_seen = true;
    gap_watch_control_mtime_ns = mtime_ns;
    gap_watch_control_size = st.st_size;

    const char* reason = "control_file";
    if (!saw_enabled || !parsed_enabled) {
        return;
    }

    next.name = parsed_name ? name_value : gap_watch_name;
    next.all_classes = parsed_all_classes ? all_classes_value :
        (gap_watch_all_classes != 0);
    next.min_bytes = parsed_min_bytes ? min_bytes_value : gap_watch_min_bytes;

    if (!enabled_value) {
        next.enabled = false;
        next.start = 0;
        next.end = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        apply_gap_watch_config_locked(next, "control_file", "disabled");
        return;
    }

    if (!parsed_start || !parsed_end || end_value < start_value) {
        next.enabled = false;
        next.start = 0;
        next.end = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        apply_gap_watch_config_locked(next, "control_file", "invalid_range");
        return;
    }

    next.enabled = true;
    next.start = start_value;
    next.end = end_value;

    std::lock_guard<std::mutex> lock(log_mutex);
    apply_gap_watch_config_locked(next, "control_file", reason);
}

/**
 * 启发式分类器：将显存申请映射到具体的业务类别
 * * 这个函数实现了从“通用内存申请”到“业务意图”的转换。
 * 它依赖于上层 Python 代码通过 uvm_set_phase() 注入的上下文。
 * * @param phase 当前运行阶段的字符串标识（由 Python 侧设置）
 * @param size  申请的内存大小（字节）
 * @return AllocationClass 推断出的内存类别
 */
static AllocationClass classify_allocation(const std::string& phase,
                                           size_t size) {
    // 1. 模型加载阶段：判定为【持久权重】
    // 这里的内存通常包含 Transformer 的各层参数（Linear/Embedding 等）。
    // 这些内存一旦分配，在整个服务生命周期内都不会释放，且每一轮推理都要全量读取。
    if (phase == "load_model") {
        return AllocationClass::WeightPersistent;
    }

    // 2. KV Cache 初始化阶段：判定为【持久缓存】
    // vLLM 通常会预先申请一个巨大的显存池（Block Manager 管理的空间）。
    // 这是 UVM 调优的重灾区，决定了最大并发量。
    if (phase == "initialize_kv_cache") {
        return AllocationClass::KvPersistent;
    }

    // 3. 预热/预处理阶段：判定为【启动工作区】
    // 包含 Autotuning（寻找算子最优配置）、Profile（试运行）等产生的临时内存。
    // 特点是：在启动时很忙，但模型正式跑起来（Inference）后基本就没用了。
    if (phase_contains(phase, "warmup") ||
        phase_contains(phase, "autotune") ||
        phase_contains(phase, "preinit") ||
        phase == "profile_run") {
        return AllocationClass::WarmupWorkspace;
    }

    // 4. 正式推理阶段：根据大小进一步细分
    // 当 phase 被设置为 "enabled" 时，代表模型已经开始处理用户请求。
    // 此时无法仅凭阶段名区分意图，必须引入【尺寸启发式 (Size Heuristics)】。
    if (phase == "enabled") {
        // 4.1 运行时临时碎片 (1MB ~ 16MB)
        // 通常是一些算子的中间极短命输出，或者很小的临时变量。
        if (size >= 1 * 1024 * 1024 && size <= 16 * 1024 * 1024) {
            return AllocationClass::RuntimeScratch;
        }
        
        // 4.2 运行时工作空间 (16MB ~ 128MB)
        // 可能是中间层的激活值 (Activations) 或 Logits。
        // 这些内存会在每一轮推理迭代中循环申请和释放。
        if (size > 16 * 1024 * 1024 && size <= 128 * 1024 * 1024) {
            return AllocationClass::RuntimeWorkspace;
        }
    }

    // 5. 兜底逻辑：无法识别的托管内存
    // 如果日志中出现大量 UnknownManaged，说明你可能发现了一个新的 vLLM 内部阶段，
    // 需要回来更新这个“分拣规则”。
    return AllocationClass::UnknownManaged;
}

/**
 * 策略选择器：决定对当前内存块执行何种管理动作
 * * 这是将“分类结果”转化为“具体指令”的转换逻辑。
 * * @param alloc_class 之前由 classify_allocation 识别出的内存类别
 * @param size        内存块大小
 * @param device      目标设备 ID（>=0 为 GPU，<0 通常指 CPU）
 * @return PolicyAction 最终决定的动作类型
 */
static PolicyAction choose_policy_action(AllocationClass alloc_class,
                                         size_t size,
                                         int device) {
    // 1. 准入检查：全局开关与设备合法性
    // 如果全局策略开关 policy_enabled 被关闭（阶段 0 常驻状态），
    // 或者目标设备是 CPU (device < 0)，则不做任何干预，直接返回默认处理。
    if (!policy_enabled || device < 0) {
        return PolicyAction::ManagedDefault;
    }

    // 2. 预热期策略触发 (Warmup Optimization)
    // 这是目前代码中唯一的“干预点”。
    // 触发条件需要同时满足以下三点：
    //   - 类别匹配：必须被识别为 WarmupWorkspace（预热期工作区）。
    //   - 功能开启：环境变量 VLLM_UVM_POLICY_WARMUP_PREFETCH 为 true。
    //   - 尺寸达标：大小必须超过阈值（policy_warmup_prefetch_min_bytes），防止频繁预取碎小内存。
    if (alloc_class == AllocationClass::WarmupWorkspace &&
        policy_warmup_prefetch_enabled &&
        size >= policy_warmup_prefetch_min_bytes) {
        
        // 如果满足，下达“预取到 GPU”的指令
        return PolicyAction::ManagedPrefetchGpu;
    }

    // 3. 默认回退
    // 对于其他类别（如 WeightPersistent 或 RuntimeScratch），
    // 目前版本的代码选择“相信”驱动程序的默认缺页调度，不主动干预。
    return PolicyAction::ManagedDefault;
}

/**
 * Initialize log file (called once on first allocation)
 */
/**
 * 初始化日志文件与全局策略配置
 * * 该函数采用延迟加载（Lazy Initialization）模式，在第一次调用 uvm_malloc 时触发。
 * 它负责从环境变量中读取用户的调优意图，并打印本次 Session 的配置清单。
 */
static void init_log_file() {
    // 1. 快速检查：如果已经初始化过，直接返回。
    // 这是为了避免在高性能路径中每次分配都要竞争锁。
    if (log_initialized) return;

    // 2. 双重检查锁定模式 (Double-Checked Locking)
    // 加锁防止多个 CPU 线程同时执行初始化逻辑。
    std::lock_guard<std::mutex> lock(log_mutex);
    
    // 获取锁后再次检查，确保在等待锁期间没有其他线程完成初始化。
    if (log_initialized) return;

    // 3. 时间基准建立
    // 记录程序启动/首次分配的时间点，后续日志中的 [+1.234s] 相对时间戳都以此为原点。
    start_time = std::chrono::steady_clock::now();

    // 4. 从环境变量读取策略配置 (Control Panel)
    // 这种设计允许研究员在不重新编译 C++ 代码的情况下，通过 Shell 直接调整 UVM 行为。
    trace_min_bytes = read_trace_min_bytes_from_env();
    
    // 默认开启策略层
    policy_enabled = read_bool_from_env("VLLM_UVM_POLICY_ENABLE", true) ? 1 : 0;
    
    // 读取策略模式（trace_only 代表演习模式，prefetch 代表实战模式）
    policy_mode = read_string_from_env("VLLM_UVM_POLICY_MODE", "trace_only");

    // 级联逻辑：如果模式被设为 prefetch 或 warmup_prefetch，
    // 则自动强制开启 policy_warmup_prefetch_enabled 开关。
    policy_warmup_prefetch_enabled = 
        policy_mode == "prefetch" || 
        policy_mode == "warmup_prefetch" || 
        read_bool_from_env("VLLM_UVM_POLICY_WARMUP_PREFETCH", false);

    // 是否开启 Preferred Location (建议位置) 提示
    policy_warmup_advise_gpu = 
        read_bool_from_env("VLLM_UVM_POLICY_WARMUP_ADVISE_GPU", false) ? 1 : 0;

    // 读取预取的尺寸下限阈值
    policy_warmup_prefetch_min_bytes = read_size_from_env(
        "VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES",
        policy_warmup_prefetch_min_bytes
    );
    unknown_detail_enabled =
        read_bool_from_env("VLLM_UVM_UNKNOWN_DETAIL_ENABLE", false) ? 1 : 0;
    unknown_detail_min_bytes = read_size_from_env(
        "VLLM_UVM_UNKNOWN_DETAIL_MIN_BYTES",
        unknown_detail_min_bytes
    );
    gap_watch_enabled =
        read_bool_from_env("VLLM_UVM_GAP_WATCH_ENABLE", false) ? 1 : 0;
    gap_watch_all_classes =
        read_bool_from_env("VLLM_UVM_GAP_WATCH_ALL_CLASSES", true) ? 1 : 0;
    gap_watch_min_bytes = read_size_from_env(
        "VLLM_UVM_GAP_WATCH_MIN_BYTES",
        gap_watch_min_bytes
    );
    gap_watch_name = read_string_from_env(
        "VLLM_UVM_GAP_WATCH_NAME",
        gap_watch_name.c_str()
    );
    gap_watch_control_file = read_string_from_env(
        "VLLM_UVM_GAP_WATCH_CONTROL_FILE",
        ""
    );
    gap_watch_refresh_ms = read_size_from_env(
        "VLLM_UVM_GAP_WATCH_REFRESH_MS",
        gap_watch_refresh_ms
    );
    uintptr_t parsed_gap_start = 0;
    uintptr_t parsed_gap_end = 0;
    if (read_hex_u64_from_env("VLLM_UVM_GAP_WATCH_START", &parsed_gap_start) &&
        read_hex_u64_from_env("VLLM_UVM_GAP_WATCH_END", &parsed_gap_end) &&
        parsed_gap_end >= parsed_gap_start) {
        gap_watch_start = parsed_gap_start;
        gap_watch_end = parsed_gap_end;
        gap_watch_enabled = 1;
    }

    // 5. 日志文件路径确定
    // 默认文件名为 vllm_uvm_allocations.log，可通过 VLLM_UVM_LOG_FILE 修改。
    const char* log_path = getenv("VLLM_UVM_LOG_FILE");
    if (!log_path) {
        log_path = "vllm_uvm_allocations.log";
    }

    // 6. 以追加模式 ("a") 打开文件
    log_file = fopen(log_path, "a");
    if (log_file) {
        char timestamp[64];
        get_timestamp(timestamp, sizeof(timestamp));

        // 7. 打印 Session 头部信息 (Session Header)
        // 这一步在科研中极其重要，因为它记录了本次实验的所有超参数。
        // 以后你回顾几个月前的日志时，一眼就能看出当时有没有开预取。
        fprintf(log_file, "\n========================================\n");
        fprintf(log_file, "[%s] vLLM UVM Allocator Session Started\n", timestamp);
        fprintf(log_file, "[%s] trace_min_bytes=%zu current_phase=%s\n", 
                timestamp, trace_min_bytes, current_phase.c_str());
        fprintf(
            log_file,
            "[%s] policy_enabled=%d policy_mode=%s "
            "policy_warmup_prefetch_enabled=%d "
            "policy_warmup_prefetch_min_bytes=%zu "
            "policy_warmup_advise_gpu=%d "
            "unknown_detail_enabled=%d unknown_detail_min_bytes=%zu "
            "gap_watch_enabled=%d gap_watch_name=%s "
            "gap_watch_start=0x%llx gap_watch_end=0x%llx "
            "gap_watch_all_classes=%d gap_watch_min_bytes=%zu "
            "gap_watch_control_file=%s gap_watch_refresh_ms=%zu\n",
            timestamp, policy_enabled, policy_mode.c_str(), 
            policy_warmup_prefetch_enabled, policy_warmup_prefetch_min_bytes, 
            policy_warmup_advise_gpu,
            unknown_detail_enabled,
            unknown_detail_min_bytes,
            gap_watch_enabled,
            gap_watch_name.c_str(),
            static_cast<unsigned long long>(gap_watch_start),
            static_cast<unsigned long long>(gap_watch_end),
            gap_watch_all_classes,
            gap_watch_min_bytes,
            gap_watch_control_file.empty() ? "none" : gap_watch_control_file.c_str(),
            gap_watch_refresh_ms
        );
        fprintf(log_file, "========================================\n");
        
        // 强制刷新缓冲区，确保头部信息立即落盘，防止程序崩溃导致日志丢失。
        fflush(log_file);
    } else {
        // 如果日志打开失败（如目录无权限），降级到 stderr 报警，但不阻塞分配流程。
        fprintf(stderr, "[vLLM UVM] Warning: Could not open log file: %s\n", log_path);
    }

    // 8. 标记初始化完成
    log_initialized = true;
    refresh_gap_watch_from_control_file_if_needed(true);
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
    uintptr_t end = region_end_for(start, size);

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
    uintptr_t end = region_end_for(start, size);
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

static void trace_policy_event(void* ptr,
                               size_t size,
                               size_t alloc_num,
                               int device,
                               const std::string& phase,
                               AllocationClass alloc_class,
                               PolicyAction action,
                               const char* size_bucket,
                               bool action_success,
                               const char* action_error) {
    if (!log_file || !policy_enabled) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    const char* safe_error = (action_error && action_error[0] != '\0') ? action_error : "none";

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_POLICY alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s action=%s action_success=%d action_error=%s\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(alloc_class),
        policy_action_to_string(action),
        action_success ? 1 : 0,
        safe_error
    );
    fflush(log_file);
}

static void trace_unknown_detail_event(void* ptr,
                                       size_t size,
                                       size_t alloc_num,
                                       int device,
                                       const std::string& phase,
                                       const char* size_bucket,
                                       PolicyAction action,
                                       cudaStream_t stream,
                                       bool overlaps_gap_watch,
                                       uintptr_t gap_overlap_start,
                                       uintptr_t gap_overlap_end,
                                       size_t gap_overlap_bytes) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_UNKNOWN_DETAIL alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_bucket=%s device=%d phase=%s predicted_class=%s "
        "action=%s stream=0x%llx gap_watch_name=%s overlaps_gap_watch=%d "
        "gap_overlap_start=0x%llx gap_overlap_end=0x%llx gap_overlap_bytes=%zu\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(AllocationClass::UnknownManaged),
        policy_action_to_string(action),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),
        gap_watch_name.c_str(),
        overlaps_gap_watch ? 1 : 0,
        static_cast<unsigned long long>(gap_overlap_start),
        static_cast<unsigned long long>(gap_overlap_end),
        gap_overlap_bytes
    );
    fflush(log_file);
}

static void trace_gap_watch_alloc_event(void* ptr,
                                        size_t size,
                                        size_t alloc_num,
                                        int device,
                                        const std::string& phase,
                                        AllocationClass alloc_class,
                                        PolicyAction action,
                                        const char* size_bucket,
                                        cudaStream_t stream,
                                        uintptr_t overlap_start,
                                        uintptr_t overlap_end,
                                        size_t overlap_bytes) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double gap_ratio = 0.0;
    if (gap_watch_enabled && gap_watch_end >= gap_watch_start) {
        size_t gap_size = gap_watch_end - gap_watch_start + 1;
        if (gap_size > 0) {
            gap_ratio = static_cast<double>(overlap_bytes) /
                        static_cast<double>(gap_size);
        }
    }

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_ALLOC alloc_id=%zu watch_name=%s "
        "ptr=0x%llx end=0x%llx size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s action=%s stream=0x%llx overlap_start=0x%llx "
        "overlap_end=0x%llx overlap_bytes=%zu overlap_ratio_of_watch=%.6f\n",
        timestamp,
        elapsed,
        alloc_num,
        gap_watch_name.c_str(),
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(alloc_class),
        policy_action_to_string(action),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),
        static_cast<unsigned long long>(overlap_start),
        static_cast<unsigned long long>(overlap_end),
        overlap_bytes,
        gap_ratio
    );
    fflush(log_file);
}

static void trace_gap_watch_free_event(void* ptr,
                                       size_t size,
                                       size_t free_num,
                                       size_t current_total,
                                       int device,
                                       const AllocationInfo* info,
                                       const std::string& current_phase_snapshot) {
    if (!log_file || !info || !info->gap_watch_logged) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double lifetime_seconds = elapsed - info->alloc_elapsed_seconds;
    double gap_ratio = 0.0;
    if (gap_watch_enabled && gap_watch_end >= gap_watch_start) {
        size_t gap_size = gap_watch_end - gap_watch_start + 1;
        if (gap_size > 0) {
            gap_ratio = static_cast<double>(info->gap_overlap_bytes) /
                        static_cast<double>(gap_size);
        }
    }

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_FREE free_id=%zu watch_name=%s "
        "ptr=0x%llx end=0x%llx size_bytes=%zu device=%d phase=%s alloc_id=%zu "
        "alloc_phase=%s alloc_predicted_class=%s alloc_action=%s "
        "overlap_start=0x%llx overlap_end=0x%llx overlap_bytes=%zu "
        "overlap_ratio_of_watch=%.6f lifetime_s=%.6f total_bytes=%zu\n",
        timestamp,
        elapsed,
        free_num,
        gap_watch_name.c_str(),
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        device,
        current_phase_snapshot.c_str(),
        info->alloc_id,
        info->phase.c_str(),
        info->alloc_class_name.c_str(),
        info->policy_action_name.c_str(),
        static_cast<unsigned long long>(info->gap_overlap_start),
        static_cast<unsigned long long>(info->gap_overlap_end),
        info->gap_overlap_bytes,
        gap_ratio,
        lifetime_seconds,
        current_total
    );
    fflush(log_file);
}

static void trace_unknown_detail_free_event(void* ptr,
                                            size_t size,
                                            size_t free_num,
                                            size_t current_total,
                                            int device,
                                            const AllocationInfo* info,
                                            const std::string& current_phase_snapshot) {
    if (!log_file || !info || !info->unknown_detail_logged) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double lifetime_seconds = elapsed - info->alloc_elapsed_seconds;

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_UNKNOWN_DETAIL_FREE free_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu device=%d phase=%s alloc_id=%zu alloc_phase=%s "
        "alloc_predicted_class=%s alloc_action=%s gap_watch_name=%s "
        "gap_overlap_bytes=%zu lifetime_s=%.6f total_bytes=%zu\n",
        timestamp,
        elapsed,
        free_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        device,
        current_phase_snapshot.c_str(),
        info->alloc_id,
        info->phase.c_str(),
        info->alloc_class_name.c_str(),
        info->policy_action_name.c_str(),
        gap_watch_name.c_str(),
        info->gap_overlap_bytes,
        lifetime_seconds,
        current_total
    );
    fflush(log_file);
}

/**
 * Allocate CUDA managed (UVM) memory
 *
 * 该函数是 vLLM/PyTorch 底层显存分配器的 Pluggable Allocator 核心入口。
 * 它的主要使命是拦截常规的 cudaMalloc，将其替换为 UVM 分配，
 * 从而允许大模型（权重+KV Cache）的显存占用突破物理 GPU 显存上限。
 *
 * @param size 请求分配的字节数
 * @param device 目标 CUDA 设备 ID
 * @param stream CUDA 流（对于纯 Managed Memory 分配本身不生效，但在后续的 Prefetch 中会用到）
 * @return 成功返回内存指针，失败返回 NULL
 */
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    // 1. 延迟初始化：确保在进程生命周期内首次分配时加载环境变量和日志配置
    if (!log_initialized) {
        init_log_file();
    }

    void* ptr = NULL;
    cudaError_t err;

    // 2. 核心分配逻辑：使用 cudaMallocManaged 申请 UVM
    // cudaMemAttachGlobal 参数至关重要，它向 CUDA 驱动声明这块内存对主机 CPU 和所有 GPU 均可访问。
    // 这使得底层驱动可以通过 Page Fault (缺页中断) 在主机 RAM 和 GPU VRAM 之间自动换页。
    err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMallocManaged failed for %zd bytes: %s\n",
                size, cudaGetErrorString(err));
        return NULL;
    }

    // 3. 维护全局原子统计信息（高并发友好）
    // fetch_add 返回的是相加之前的值，所以加上 size 才是当前实际值
    size_t current = total_allocated.fetch_add(size) + size;
    size_t alloc_count = num_allocs.fetch_add(1) + 1;

    // 4. 无锁 (Lock-free) 更新显存高水位线 (Peak Memory)
    // 使用 Compare-And-Swap (CAS) 循环来处理多线程并发分配时的竞争状态，
    // 确保 peak_allocated 准确记录历史最高分配量。
    size_t peak = peak_allocated.load();
    while (current > peak) {
        if (peak_allocated.compare_exchange_weak(peak, current)) {
            break;
        }
    }

    std::string phase_snapshot;
    double alloc_elapsed = get_elapsed_seconds();
    
    // 5. 捕获当前 phase 快照
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;
    }

    // 6. 分级日志记录
    // 对于超大块内存 (> 100MB，通常是模型权重加载或大规模 KV Cache 初始化)，强制记录简报
    if (size > 100 * 1024 * 1024) {
        log_allocation("ALLOC", size, alloc_count, current, peak_allocated.load(), device);
    }

    // 结构化的详细追踪日志，用于后续的离线分析或 eBPF 性能重放关联
    if (size >= trace_min_bytes) {
        trace_allocation_event(
            ptr, static_cast<size_t>(size), alloc_count, current, 
            peak_allocated.load(), device, phase_snapshot
        );
    }

    // 标准错误输出 (stderr) 回显，通常用于本地调试
    if (verbose_logging && size > 100 * 1024 * 1024) {
        fprintf(stderr, "[vLLM UVM] Alloc #%zu: %.2f MB (total: %.2f GB, peak: %.2f GB)\n",
                alloc_count, size / 1e6, current / 1e9,
                peak_allocated.load() / 1e9);
    }

    // 7. 启发式内存策略推断 (Heuristic Policy Engine)
    // 结合当前运行阶段 (如 warmup) 和申请大小，推测这块内存的用途
    AllocationClass alloc_class = classify_allocation(
        phase_snapshot,
        static_cast<size_t>(size)
    );
    // 决定是否要对该内存块采取特殊动作（如主动 Advise 或 Prefetch）
    PolicyAction policy_action = choose_policy_action(
        alloc_class,
        static_cast<size_t>(size),
        device
    );

    refresh_gap_watch_from_control_file_if_needed(false);
    
    const char* policy_action_error = "none";
    bool policy_action_success = true;
    const char* size_bucket = size_bucket_for(static_cast<size_t>(size));
    uintptr_t gap_overlap_start = 0;
    uintptr_t gap_overlap_end = 0;
    size_t gap_overlap_bytes = 0;
    bool overlaps_gap_watch = compute_overlap(
        reinterpret_cast<uintptr_t>(ptr),
        static_cast<size_t>(size),
        gap_watch_start,
        gap_watch_end,
        &gap_overlap_start,
        &gap_overlap_end,
        &gap_overlap_bytes
    );
    bool log_unknown_detail = should_trace_unknown_detail(
        alloc_class,
        static_cast<size_t>(size)
    );
    bool log_gap_watch = should_trace_gap_watch(
        alloc_class,
        static_cast<size_t>(size),
        gap_overlap_bytes
    );
    bool store_active_info =
        static_cast<size_t>(size) >= trace_min_bytes ||
        log_unknown_detail || log_gap_watch;

    // 7.1 记录活跃分配元数据，供 free / gap watch / unknown detail 使用
    if (store_active_info) {
        std::lock_guard<std::mutex> lock(log_mutex);
        active_allocations[ptr] = AllocationInfo{
            static_cast<size_t>(size),
            device,
            alloc_count,
            phase_snapshot,
            alloc_elapsed,
            allocation_class_to_string(alloc_class),
            policy_action_to_string(policy_action),
            size_bucket,
            log_unknown_detail,
            log_gap_watch,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes,
        };
    }

    // 8. 策略执行阶段：主动显存调优
    if (policy_action == PolicyAction::ManagedPrefetchGpu && ptr != NULL) {
        
        // 8.1 内存建议 (MemAdvise)：提示 CUDA 驱动该内存的首选物理位置在 GPU 上
        // 这可以在真正发生缺页中断时，降低驱动寻找最佳页存放地的开销。
        if (policy_warmup_advise_gpu) {
            cudaError_t advise_err = cudaMemAdvise(
                ptr,
                static_cast<size_t>(size),
                cudaMemAdviseSetPreferredLocation,
                device
            );
            if (advise_err != cudaSuccess) {
                policy_action_success = false;
                policy_action_error = cudaGetErrorString(advise_err);
            }
        }

        // 8.2 异步预取 (Prefetch Async)
        // 主动将内存页迁移到 GPU 上，避免首次访问时触发昂贵的 Page Fault。
        // 工程考量：大规模的 GPU 跨块预取 (Cross-block prefetch) 极易引发 PCIe 带宽被严重挤占，
        // 反而会导致其他并发的推理任务因总线竞争而性能劣化。
        // 因此这里被严格约束在特定 policy（如专门的 warmup 阶段或极特定的内存类）下才触发。
        cudaError_t prefetch_err = cudaMemPrefetchAsync(
            ptr,
            static_cast<size_t>(size),
            device,
            stream // 绑定到特定流，允许与该流上的计算重叠
        );
        if (prefetch_err != cudaSuccess) {
            policy_action_success = false;
            policy_action_error = cudaGetErrorString(prefetch_err);
        }
    }

    // 记录策略引擎的判定与执行结果
    if (policy_enabled) {
        trace_policy_event(
            ptr, static_cast<size_t>(size), alloc_count, device,
            phase_snapshot, alloc_class, policy_action,
            size_bucket,
            policy_action_success, policy_action_error
        );
    }

    if (log_unknown_detail) {
        trace_unknown_detail_event(
            ptr,
            static_cast<size_t>(size),
            alloc_count,
            device,
            phase_snapshot,
            size_bucket,
            policy_action,
            stream,
            overlaps_gap_watch,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes
        );
    }
    if (log_gap_watch) {
        trace_gap_watch_alloc_event(
            ptr,
            static_cast<size_t>(size),
            alloc_count,
            device,
            phase_snapshot,
            alloc_class,
            policy_action,
            size_bucket,
            stream,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes
        );
    }

    // 9. 全局暴力预取 Fallback (通常默认关闭)
    // 如果系统开启了全局预取 (enable_prefetch=1)，且策略引擎没有覆盖该内存块，则执行普适性预取。
    // 在生产环境的推理服务中，由于 PCIe 带宽的脆弱性，此选项通常不建议保持常开。
    if (enable_prefetch && device >= 0 && ptr != NULL &&
        policy_action != PolicyAction::ManagedPrefetchGpu) {
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
        if (has_info) {
            trace_unknown_detail_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                current,
                device,
                &info,
                phase_snapshot
            );
            trace_gap_watch_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                current,
                device,
                &info,
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
