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
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <cctype>
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
static std::string gap_watch_target_class = "any";
static std::string gap_watch_policy_action = "observe";
static std::string gap_watch_control_file = "";
static size_t gap_watch_refresh_ms = 250;
static int device_direct_enable = 0;
static size_t device_direct_min_bytes = 4096;
static size_t device_direct_max_bytes = 1 * 1024 * 1024;
static size_t device_direct_max_total_bytes = 256 * 1024 * 1024;
static std::string device_direct_backend = "cuda_malloc";
static std::string device_direct_target_phases =
    "enabled:attention,enabled:moe,enabled:model_forward";
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

// =============================================================================
// 全局策略遥测与效能评估指标 (Global Policy Telemetry Metrics)
// 采用无锁 (Lock-free) 的 std::atomic 实现，确保在多流、多线程高并发推理场景下
// 收集指标时不会引入额外的锁竞争与长尾延迟。这些数据通常在 Session 结束时汇总打印。
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Gap Watch 基础观测指标 (Gap Watch Observation Metrics)
// 用于量化目标虚拟/物理地址空间（Gap）的内存活动密集度（即：热点有多“热”）
// -----------------------------------------------------------------------------

/// @brief 触发 Gap 重叠的分配总次数。只要申请的内存块与监控区间有交集即 +1。
static std::atomic<size_t> gap_watch_overlap_allocs{0};

/// @brief 落入监控区间的实际重叠字节总量。用于计算“热点区域空间利用率/重叠率”。
static std::atomic<size_t> gap_watch_overlap_bytes_total{0};

/// @brief 既与监控区间重叠，又完全符合预设意图类别（Target Class，如 KV Cache）的分配次数。
/// @note 该指标与 gap_watch_overlap_allocs 的差值，反映了目标热点区域内的“内存噪声”
/// （即有多少非目标类型的零碎内存恰好也分配在了这个区间）。
static std::atomic<size_t> gap_watch_target_class_match_allocs{0};


// -----------------------------------------------------------------------------
// 2. Gap Watch 策略执行指标 (Gap Watch Policy Execution Metrics)
// 用于评估主动干预动作（如强制 Advise 或 Prefetch）的覆盖面与系统兼容性
// -----------------------------------------------------------------------------

/// @brief 成功触发并尝试执行主动策略干预（提权动作）的分配次数。
static std::atomic<size_t> gap_watch_policy_applied_allocs{0};

/// @brief 被主动策略干预所覆盖的热点字节总量。
/// @note 这是评估“预取/迁移带宽开销”的核心基准数据。
static std::atomic<size_t> gap_watch_policy_applied_overlap_bytes{0};

/// @brief 底层 CUDA API（如 cudaMemPrefetchAsync）调用成功的次数。
static std::atomic<size_t> gap_watch_policy_success_allocs{0};

/// @brief 策略执行失败的次数（例如：设备不支持并发预取、CUDA 流错误等）。
/// @note 如果此数值异常升高，提示底层驱动或环境配置存在瓶颈，需结合 trace 日志排查。
static std::atomic<size_t> gap_watch_policy_failed_allocs{0};


// -----------------------------------------------------------------------------
// 3. Device Direct 演进指标 (Device Direct Evolution Metrics)
// 用于支撑系统从“纯 UVM 托管”向“混合显存池（Bypass UVM）”演进的决策依据。
// 这些是前瞻性评估指标，量化了如果切断 CPU 侧的 Page Fault 关联，能拯救多少内存。
// -----------------------------------------------------------------------------

/// @brief 参与 Device Direct 旁路分配评估的分配总次数（即经过 Device Direct Trace 策略判定的次数）。
static std::atomic<size_t> device_direct_trace_allocs{0};

/// @brief 经策略引擎判定，完全具备“安全绕过 UVM 直接分配纯显存”资格的分配次数。
/// @note 达标条件通常包括：无 CPU 访问风险、属于明确的计算临时空间 (RuntimeWorkspace)、大小在设定阈值内。
static std::atomic<size_t> device_direct_eligible_allocs{0};

/// @brief Stage C 真正路由到 GPU-only backend 的分配次数。
static std::atomic<size_t> device_direct_actual_allocs{0};

/// @brief Stage C 真正路由到 GPU-only backend 的累计字节数。
static std::atomic<size_t> device_direct_actual_bytes{0};

/// @brief Stage C GPU-only backend 当前仍然存活的真实显存字节数。
static std::atomic<size_t> device_direct_live_bytes{0};

/// @brief Stage C GPU-only backend 存活显存高水位，用于验证 C1 总预算是否生效。
static std::atomic<size_t> device_direct_peak_live_bytes{0};

/// @brief Stage C 因总预算不足而拒绝 device_direct、回退 managed 的次数。
static std::atomic<size_t> device_direct_budget_rejects{0};

/// @brief Stage C 试图启用 GPU-only backend 但 cudaMalloc 失败、回退到 managed 的次数。
static std::atomic<size_t> device_direct_fallback_allocs{0};

/// @brief Stage C GPU-only backend 分配在 free 阶段成功释放的次数。
static std::atomic<size_t> device_direct_free_success_allocs{0};

/// @brief 所有参与评估的内存请求累计字节量。
/// @note 结合 eligible_allocs 的大小，可以量化出下一阶段（Phase C）真正开启 Device Direct 后，
/// 能为系统节省多少 UVM 缺页追踪开销。
static std::atomic<size_t> device_direct_requested_bytes{0};

/**
 * @struct AllocationInfo
 * @brief UVM 显存分配的元数据生命周期快照 (Metadata Lifecycle Snapshot)
 * * 此结构体充当每个 UVM 显存块的“数字档案”。它在内存分配 (cudaMallocManaged) 时被创建，
 * 记录了从底层物理属性到上层业务意图（如 KV Cache、Weights）的完整映射关系。
 * 这些详尽的追踪数据不仅用于运行时的动态策略干预（Gap Watch/Device Direct），
 * 更是离线分析 CPU-GPU 耦合瓶颈、生成 eBPF 性能重放日志的核心数据源。
 */
struct AllocationInfo {
    // ========================================================================
    // 1. 基础物理与生命周期元数据 (Base Physical & Lifecycle Metadata)
    // ========================================================================

    size_t size;                    ///< 分配的内存字节数。
    int device;                     ///< 目标设备的 ID（>=0 为特定 GPU，-1 通常代表 CPU/主机内存）。
    size_t alloc_id;                ///< 全局单调递增的分配序号，用于在海量日志中精确定位单次申请。
    std::string phase;              ///< 分配发生时上层系统（如 vLLM）所处的业务阶段（例如 "load_model", "initialize_kv_cache"）。
    double alloc_elapsed_seconds;   ///< 相对于调度器/拦截器启动时间的时间戳。配合 free 时的相对时间，可精确计算内存生命周期。
    std::string size_bucket;        ///< 将离散的 size 映射到预定义的区间（如 "1MiB-2MiB"），方便进行显存碎片率或分配频率的直方图统计。

    // ========================================================================
    // 2. 启发式意图与策略引擎状态 (Heuristic Intent & Policy Engine State)
    // ========================================================================

    std::string alloc_class_name;   ///< 启发式分类器推断出的内存用途类别（如 "WeightPersistent", "RuntimeScratch"）。
    std::string policy_action_name; ///< 最终决定执行的主动干预动作（如 "ManagedPrefetchGpu", "ManagedDefault"）。
    std::string policy_source_name; ///< 触发该动作的策略来源（"base_policy" 默认启发式，或 "gap_watch_policy" 动态热点干预）。
    bool policy_action_success;     ///< 记录 cudaMemAdvise 或 cudaMemPrefetchAsync 是否调用成功。
    std::string policy_action_error;///< 如果策略执行失败（如流冲突或设备不支持），记录具体的 CUDA 错误字符串。

    // ========================================================================
    // 3. 动态热点观测与隔离 (Dynamic Gap Watch & Hotspot Tracking)
    // ========================================================================

    bool unknown_detail_logged;     ///< 标志位：是否已将此未知内存的详细信息落盘，防止日志风暴。
    bool gap_watch_logged;          ///< 标志位：是否已被 Gap Watch 机制捕获并记录。
    uintptr_t gap_overlap_start;    ///< 该内存块与目标观测区间 (Gap) 发生重叠的起始绝对虚拟地址。
    uintptr_t gap_overlap_end;      ///< 重叠区域的结束地址。
    size_t gap_overlap_bytes;       ///< 实际落在观测热点区间内的字节数，用于计算重叠率 (Overlap Ratio)。
    std::string gap_watch_target_class_name;  ///< 触发监控时，Gap Watch 所关注的目标内存类别。
    std::string gap_watch_policy_action_name; ///< Gap Watch 机制为此热点区域尝试执行的提权动作（如强制预取）。
    bool hot_gap_match;             ///< 核心高频标志：该分配是否完美命中了当前配置的活跃热点区域。

    // ========================================================================
    // 4. 后端路由与直接内存分配演进 (Backend Routing & Device Direct Stage)
    // 用于量化 CPU-GPU 耦合问题，并在必要时绕过 UVM 直接分配纯显存
    // ========================================================================

    std::string placement_backend_name; ///< 实际使用的分配后端（当前大多为 "managed"，为演进到 "device_only" 预留）。
    std::string device_direct_backend_name; ///< device_direct 实际使用的 CUDA 分配后端，如 cuda_malloc 或 cuda_malloc_async。
    bool device_direct_eligible;    ///< 评估此内存块是否具备绕过 UVM、直接使用 cudaMalloc 的资格（取决于大小和生命周期）。
    std::string device_direct_reason; ///< 判定是否具备绕过资格的具体原因（如 "below_min_bytes", "target_class_mismatch"）。
    std::string cpu_access_risk;    ///< 风险评估标签：如果将其强制放置在 GPU 上，未来被 CPU 访问触发严重 Page Fault 的风险等级。
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
    ManagedAdvisePrefetchGpu,//gap watch 的更激进版本：advise + prefetch
    DeviceDirectTrace,//阶段 B：只记录 would-use-device-direct，不改变分配行为
    DeviceDirect,//阶段 B 默认仍 trace-only，阶段 C 才会真正切换 GPU-only backend
};

static const char* allocation_class_to_string(AllocationClass alloc_class);

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

static size_t read_size_from_env_allow_zero(const char* name, size_t default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0') {
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

static std::string lower_copy(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

static bool string_equals_ignore_case(const std::string& left,
                                      const std::string& right) {
    return lower_copy(left) == lower_copy(right);
}

static bool phase_contains(const std::string& phase, const char* needle) {
    return phase.find(needle) != std::string::npos;
}

static bool phase_starts_with(const std::string& phase,
                              const std::string& prefix) {
    return phase.rfind(prefix, 0) == 0;
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

static std::string normalize_device_direct_backend(const std::string& value) {
    std::string lowered = lower_copy(trim_copy(value));
    if (lowered == "cuda_malloc_async" || lowered == "cuda_async") {
        return "cuda_malloc_async";
    }
    return "cuda_malloc";
}

static bool comma_list_contains_phase_prefix(const std::string& csv,
                                             const std::string& phase) {
    if (csv.empty() || string_equals_ignore_case(csv, "any")) {
        return true;
    }

    size_t start = 0;
    while (start <= csv.size()) {
        size_t comma = csv.find(',', start);
        std::string token = trim_copy(
            csv.substr(
                start,
                comma == std::string::npos ? std::string::npos : comma - start
            )
        );
        if (!token.empty() && phase_starts_with(phase, token)) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}

static bool is_device_direct_target_phase(const std::string& phase) {
    return comma_list_contains_phase_prefix(device_direct_target_phases, phase);
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

static bool parse_policy_action_string(const std::string& value,
                                       PolicyAction* out) {
    if (!out) {
        return false;
    }
    std::string lowered = lower_copy(value);
    if (lowered == "observe" || lowered == "managed_default") {
        *out = PolicyAction::ManagedDefault;
        return true;
    }
    if (lowered == "prefetch" || lowered == "managed_prefetch_gpu") {
        *out = PolicyAction::ManagedPrefetchGpu;
        return true;
    }
    if (lowered == "advise_prefetch" ||
        lowered == "managed_advise_prefetch_gpu") {
        *out = PolicyAction::ManagedAdvisePrefetchGpu;
        return true;
    }
    if (lowered == "device_direct_trace") {
        *out = PolicyAction::DeviceDirectTrace;
        return true;
    }
    if (lowered == "device_direct") {
        *out = PolicyAction::DeviceDirect;
        return true;
    }
    return false;
}

static bool gap_watch_target_class_matches(AllocationClass alloc_class,
                                           const std::string& phase) {
    const char* alloc_class_name = allocation_class_to_string(alloc_class);
    if (!alloc_class_name || alloc_class_name[0] == '\0') {
        return false;
    }
    if (gap_watch_target_class.empty() ||
        string_equals_ignore_case(gap_watch_target_class, "any")) {
        return true;
    }
    if (string_equals_ignore_case(gap_watch_target_class, "gap_hot_runtime_scratch")) {
        return is_device_direct_target_phase(phase) &&
               (alloc_class == AllocationClass::UnknownManaged ||
                alloc_class == AllocationClass::RuntimeScratch ||
                alloc_class == AllocationClass::RuntimeWorkspace);
    }
    return string_equals_ignore_case(gap_watch_target_class, alloc_class_name);
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
        case PolicyAction::DeviceDirect:
            return "device_direct";
        case PolicyAction::DeviceDirectTrace:
            return "device_direct_trace";
        case PolicyAction::ManagedAdvisePrefetchGpu:
            return "managed_advise_prefetch_gpu";
        case PolicyAction::ManagedPrefetchGpu:
            return "managed_prefetch_gpu";
        case PolicyAction::ManagedDefault:
        default:
            return "managed_default";
    }
}

static bool is_device_direct_action(PolicyAction action) {
    return action == PolicyAction::DeviceDirectTrace ||
           action == PolicyAction::DeviceDirect;
}

static void update_device_direct_peak_live(size_t current_live) {
    size_t peak = device_direct_peak_live_bytes.load();
    while (current_live > peak) {
        if (device_direct_peak_live_bytes.compare_exchange_weak(peak, current_live)) {
            break;
        }
    }
}

static bool reserve_device_direct_budget(size_t size) {
    size_t live = device_direct_live_bytes.load();
    while (true) {
        if (device_direct_max_total_bytes > 0 &&
            live > device_direct_max_total_bytes - std::min(size, device_direct_max_total_bytes)) {
            return false;
        }
        size_t next_live = live + size;
        if (next_live < live) {
            return false;
        }
        if (device_direct_max_total_bytes > 0 &&
            next_live > device_direct_max_total_bytes) {
            return false;
        }
        if (device_direct_live_bytes.compare_exchange_weak(live, next_live)) {
            update_device_direct_peak_live(next_live);
            return true;
        }
    }
}

static void release_device_direct_budget(size_t size) {
    size_t live = device_direct_live_bytes.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (device_direct_live_bytes.compare_exchange_weak(live, next_live)) {
            return;
        }
    }
}

static size_t device_direct_budget_remaining_snapshot(size_t live) {
    if (device_direct_max_total_bytes == 0) {
        return 0;
    }
    return live >= device_direct_max_total_bytes
        ? 0
        : device_direct_max_total_bytes - live;
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

/**
 * @struct GapWatchConfigSnapshot
 * @brief 动态显存热点观测配置的“不可变快照” (Immutable Snapshot of Gap Watch Configuration)
 * * @details
 * 该结构体充当 Gap Watch 机制的控制平面 (Control Plane) 与数据平面 (Data Plane) 之间的契约。
 * 系统通过轮询控制文件 (Control File)，在后台将最新的监控意图打包成此快照。
 * 底层的 `uvm_malloc` / `uvm_free` 会直接基于此快照进行 O(1) 复杂度的重叠计算与策略路由，
 * 从而在实现“零侵入式策略热更新”的同时，避免在 CUDA 分配关键路径上引入严重的锁竞争。
 */
struct GapWatchConfigSnapshot {
    // ========================================================================
    // 1. 空间定位与全局开关 (Spatial Targeting & Master Switch)
    // ========================================================================

    /// @brief 全局拨测开关。若为 false，分配器将完全跳过地址重叠计算，以最低开销放行。
    bool enabled;

    /// @brief 监控区间的起始绝对地址（通常为虚拟地址）。
    /// @note 在 xCoord 等软硬协同架构中，这可能对应一段被频繁触发 CPU-GPU UVM 缺页抖动的热点物理内存映射区。
    uintptr_t start;

    /// @brief 监控区间的结束绝对地址。与 start 共同划定了一个“楚河汉界”。
    uintptr_t end;

    /// @brief 该观测任务的可读标识符（如 "layer_0_kv_cache_hotspot"）。
    /// @note 极具工程价值：它会被直接透传到详细日志中，方便与 eBPF 探针收集到的内核级
    /// page fault 堆栈或 SysAI-Agent 的时序图谱进行无缝联合关联 (Join)。
    std::string name;

    // ========================================================================
    // 2. 启发式意图过滤器 (Heuristic Intent Filters)
    // 决定哪些内存请求有资格触发干预，避免“误杀”或陷入细碎内存的追踪泥潭。
    // ========================================================================

    /// @brief 泛类型匹配标志位。若为 true，则无视内存的具体业务语义（Class），拦截区间内所有分配。
    bool all_classes;

    /// @brief 尺寸噪声过滤器。只有单次申请大小 >= 此阈值（如 1MB）的内存块才会被监控。
    /// @note 过滤掉极高频但无足轻重的 RuntimeScratch 碎片，是保障分析信噪比的关键。
    size_t min_bytes;

    /// @brief 精准打击目标类别（当 all_classes = false 时生效）。
    /// @note 例如仅针对 "KvPersistent" (KV Cache) 执行特定的监控或迁移操作，
    /// 从而保护 "WeightPersistent" (只读权重) 不受影响。
    std::string target_class;

    // ========================================================================
    // 3. 提权干预动作 (Privilege Escalation Action)
    // ========================================================================

    /// @brief 命中观测区间且通过过滤器后，分配器需要强制重载（Override）的策略动作名称。
    /// @note 例如，原启发式规则认为这块内存该走 "ManagedDefault"，但快照指示
    /// 这里是高危抖动区，强制执行 "ManagedAdvisePrefetchGpu" 或尝试 "DeviceDirect"。
    std::string policy_action_name;
};

static GapWatchConfigSnapshot current_gap_watch_config_snapshot() {
    return GapWatchConfigSnapshot{
        gap_watch_enabled != 0,
        gap_watch_start,
        gap_watch_end,
        gap_watch_name,
        gap_watch_all_classes != 0,
        gap_watch_min_bytes,
        gap_watch_target_class,
        gap_watch_policy_action,
    };
}

static bool gap_watch_configs_equal(const GapWatchConfigSnapshot& left,
                                    const GapWatchConfigSnapshot& right) {
    return left.enabled == right.enabled &&
           left.start == right.start &&
           left.end == right.end &&
           left.name == right.name &&
           left.all_classes == right.all_classes &&
           left.min_bytes == right.min_bytes &&
           left.target_class == right.target_class &&
           left.policy_action_name == right.policy_action_name;
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
        "min_bytes=%zu target_class=%s policy_action=%s reason=%s\n",
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
        config.target_class.c_str(),
        config.policy_action_name.c_str(),
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
    gap_watch_target_class =
        config.target_class.empty() ? "any" : config.target_class;
    gap_watch_policy_action =
        config.policy_action_name.empty() ? "observe" : config.policy_action_name;

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

/**
 * @brief 从外部控制文件动态刷新 Gap Watch 配置 (Dynamic Configuration Hot-reload)
 * * @details
 * 这是一个为了在不重启 vLLM/推理服务的前提下，动态调整 UVM 观测热点而设计的轮询引擎。
 * 由于此函数在 `uvm_malloc` 的关键路径中被调用，必须极力避免长尾延迟。
 * 设计上采用了“三级短路机制”来降低 IO 与系统调用开销：
 * 1. 软限流 (Time-based Throttling)：基于时钟，避免 ms 级别内的频繁 stat 调用。
 * 2. 元数据比对 (Stat Mtime Check)：对比文件修改时间和大小，内容未变则跳过解析。
 * 3. 内存快照 (Atomic Snapshot)：仅在配置真正变更时，才加锁更新全局状态。
 * * @param force 是否无视时间节流阀，强制执行一次文件系统探测（通常在初始化阶段置为 true）。
 */
static void refresh_gap_watch_from_control_file_if_needed(bool force) {
    // 短路检查 0：如果没有配置控制文件路径，直接退回默认状态
    if (gap_watch_control_file.empty()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();

    // ========================================================================
    // 阶段 1：软限流机制 (Time-based Throttling)
    // 防止高并发显存申请时，每个线程都疯狂发起文件系统调用，导致内核态开销激增。
    // ========================================================================
    if (!force &&
        gap_watch_last_refresh_check != std::chrono::steady_clock::time_point::min()) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - gap_watch_last_refresh_check
        ).count();
        // 只有当距离上次检查的时间超过 gap_watch_refresh_ms (如 250ms) 时，才放行
        if (elapsed_ms >= 0 &&
            static_cast<size_t>(elapsed_ms) < gap_watch_refresh_ms) {
            return;
        }
    }
    // 更新最后检查时间戳（即使后续 stat 失败也需更新，避免密集重试）
    gap_watch_last_refresh_check = now;

    // ========================================================================
    // 阶段 2：文件元数据探测 (Filesystem Stat Fast-path)
    // ========================================================================
    struct stat st {};
    if (stat(gap_watch_control_file.c_str(), &st) != 0) {
        return; // 文件不存在或无权限，静默返回，保持现有配置
    }

    uint64_t mtime_ns = stat_mtime_ns(st);
    // 比对文件的 MTime 和 Size。只有当文件被实际修改过时，才进入昂贵的 fopen 和解析逻辑。
    if (!force && gap_watch_control_seen &&
        mtime_ns == gap_watch_control_mtime_ns &&
        st.st_size == gap_watch_control_size) {
        return;
    }

    // ========================================================================
    // 阶段 3：解析控制平面指令 (Control Plane Parsing)
    // ========================================================================
    FILE* control = fopen(gap_watch_control_file.c_str(), "r");
    if (!control) {
        return;
    }

    // 基于当前活跃的快照作为基准进行修改，保证未在文件中显式声明的字段继承旧值
    GapWatchConfigSnapshot next = current_gap_watch_config_snapshot();
    next.enabled = false;
    next.start = 0;
    next.end = 0;

    // 解析状态机变量
    bool saw_enabled = false;
    bool parsed_enabled = false;
    bool parsed_all_classes = false;
    bool parsed_name = false;
    bool parsed_start = false;
    bool parsed_end = false;
    bool parsed_min_bytes = false;
    bool parsed_target_class = false;
    bool parsed_policy_action = false;

    // 解析缓冲变量
    bool enabled_value = false;
    bool all_classes_value = gap_watch_all_classes != 0;
    uintptr_t start_value = 0;
    uintptr_t end_value = 0;
    size_t min_bytes_value = gap_watch_min_bytes;
    std::string name_value = gap_watch_name;
    std::string target_class_value = gap_watch_target_class;
    std::string policy_action_value = gap_watch_policy_action;
    char buffer[1024];

    // 逐行解析 K-V 格式的配置文件，支持 # 注释和空行
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

        // 路由解析指令到具体的转换函数 (支持 hex 地址、bool、size 等)
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
        } else if (key == "target_class") {
            target_class_value = value;
            parsed_target_class = true;
        } else if (key == "policy_action") {
            PolicyAction action{};
            parsed_policy_action = parse_policy_action_string(value, &action);
            if (parsed_policy_action) {
                policy_action_value = policy_action_to_string(action);
            }
        }
    }
    fclose(control);

    // 记录本次成功解析的元数据特征，用于下一次的短路检查
    gap_watch_control_seen = true;
    gap_watch_control_mtime_ns = mtime_ns;
    gap_watch_control_size = st.st_size;

    const char* reason = "control_file";
    // 强制校验：控制文件必须至少包含清晰的 enabled 字段，否则视为无效修改，丢弃变更
    if (!saw_enabled || !parsed_enabled) {
        return;
    }

    // ========================================================================
    // 阶段 4：配置快照构建与应用 (Snapshot Construction & Commit)
    // ========================================================================

    // 合并解析到的新值与系统旧值
    next.name = parsed_name ? name_value : gap_watch_name;
    next.all_classes = parsed_all_classes ? all_classes_value :
        (gap_watch_all_classes != 0);
    next.min_bytes = parsed_min_bytes ? min_bytes_value : gap_watch_min_bytes;
    next.target_class = parsed_target_class ? target_class_value :
        gap_watch_target_class;
    next.policy_action_name = parsed_policy_action ? policy_action_value :
        gap_watch_policy_action;

    // 场景 A：被用户手动置为 disabled，清空监控区间并提交
    if (!enabled_value) {
        next.enabled = false;
        next.start = 0;
        next.end = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        apply_gap_watch_config_locked(next, "control_file", "disabled");
        return;
    }

    // 场景 B：非法地址区间（如 end < start），自动执行熔断并降级为 disabled
    if (!parsed_start || !parsed_end || end_value < start_value) {
        next.enabled = false;
        next.start = 0;
        next.end = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        apply_gap_watch_config_locked(next, "control_file", "invalid_range");
        return;
    }

    // 场景 C：合法配置，正式装载新的监控边界
    next.enabled = true;
    next.start = start_value;
    next.end = end_value;

    // 获取日志与配置更新锁，原子性地提交给全局数据平面
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
    if (phase == "enabled" || phase.rfind("enabled:", 0) == 0) {
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

static PolicyAction choose_gap_watch_policy_action(AllocationClass alloc_class,
                                                   const std::string& phase,
                                                   size_t size,
                                                   int device,
                                                   size_t gap_overlap_bytes,
                                                   bool* class_match_out,
                                                   const char** source_out) {
    if (class_match_out) {
        *class_match_out = false;
    }
    if (source_out) {
        *source_out = "base_policy";
    }

    if (!policy_enabled || device < 0 || gap_overlap_bytes == 0 || !gap_watch_enabled) {
        return PolicyAction::ManagedDefault;
    }

    bool class_match = gap_watch_target_class_matches(alloc_class, phase);
    if (class_match_out) {
        *class_match_out = class_match;
    }
    if (!class_match) {
        return PolicyAction::ManagedDefault;
    }

    if (size < gap_watch_min_bytes) {
        return PolicyAction::ManagedDefault;
    }

    PolicyAction action = PolicyAction::ManagedDefault;
    if (!parse_policy_action_string(gap_watch_policy_action, &action)) {
        return PolicyAction::ManagedDefault;
    }

    if (source_out) {
        *source_out = "gap_watch_policy";
    }
    return action;
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

    {
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
        gap_watch_target_class = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_TARGET_CLASS",
            gap_watch_target_class.c_str()
        );
        gap_watch_policy_action = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_POLICY_ACTION",
            gap_watch_policy_action.c_str()
        );
        gap_watch_control_file = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_CONTROL_FILE",
            ""
        );
        gap_watch_refresh_ms = read_size_from_env(
            "VLLM_UVM_GAP_WATCH_REFRESH_MS",
            gap_watch_refresh_ms
        );
        device_direct_enable =
            read_bool_from_env("VLLM_UVM_DEVICE_DIRECT_ENABLE", false) ? 1 : 0;
        device_direct_min_bytes = read_size_from_env(
            "VLLM_UVM_DEVICE_DIRECT_MIN_BYTES",
            device_direct_min_bytes
        );
        device_direct_max_bytes = read_size_from_env(
            "VLLM_UVM_DEVICE_DIRECT_MAX_BYTES",
            device_direct_max_bytes
        );
        device_direct_max_total_bytes = read_size_from_env(
            "VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES",
            device_direct_max_total_bytes
        );
        device_direct_max_total_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES",
            device_direct_max_total_bytes
        );
        device_direct_backend = normalize_device_direct_backend(
            read_string_from_env(
                "VLLM_UVM_DEVICE_DIRECT_BACKEND",
                device_direct_backend.c_str()
            )
        );
        device_direct_target_phases = read_string_from_env(
            "VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES",
            device_direct_target_phases.c_str()
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
                "gap_watch_target_class=%s gap_watch_policy_action=%s "
                "gap_watch_control_file=%s gap_watch_refresh_ms=%zu "
                "device_direct_enable=%d device_direct_min_bytes=%zu "
                "device_direct_max_bytes=%zu "
                "device_direct_max_total_bytes=%zu "
                "device_direct_backend=%s "
                "device_direct_target_phases=%s\n",
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
                gap_watch_target_class.c_str(),
                gap_watch_policy_action.c_str(),
                gap_watch_control_file.empty() ? "none" : gap_watch_control_file.c_str(),
                gap_watch_refresh_ms,
                device_direct_enable,
                device_direct_min_bytes,
                device_direct_max_bytes,
                device_direct_max_total_bytes,
                device_direct_backend.c_str(),
                device_direct_target_phases.c_str()
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
    }

    // 注意：这里必须在释放 log_mutex 后再去 refresh 控制文件。
    // 否则 refresh_gap_watch_from_control_file_if_needed() 内部再次尝试获取
    // log_mutex 时会发生自锁，导致服务卡在 "UVM allocator enabled successfully" 之后。
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
                               const char* policy_source,
                               bool gap_watch_class_match,
                               size_t gap_overlap_bytes,
                               const char* size_bucket,
                               const char* placement_backend,
                               const char* device_direct_backend_used,
                               bool device_direct_eligible,
                               const char* device_direct_reason,
                               const char* cpu_access_risk,
                               bool hot_gap_match,
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
    size_t device_direct_live_snapshot = device_direct_live_bytes.load();
    size_t device_direct_budget_remaining =
        device_direct_budget_remaining_snapshot(device_direct_live_snapshot);

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_POLICY alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s action=%s policy_source=%s "
        "gap_watch_class_match=%d gap_overlap_bytes=%zu "
        "action_success=%d action_error=%s placement_backend=%s "
        "device_direct_backend=%s "
        "device_direct_eligible=%d device_direct_reason=%s "
        "device_direct_live_bytes=%zu device_direct_max_total_bytes=%zu "
        "device_direct_budget_remaining=%zu "
        "cpu_access_risk=%s hot_gap_match=%d\n",
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
        (policy_source && policy_source[0] != '\0') ? policy_source : "unknown",
        gap_watch_class_match ? 1 : 0,
        gap_overlap_bytes,
        action_success ? 1 : 0,
        safe_error,
        placement_backend ? placement_backend : "managed",
        device_direct_backend_used ? device_direct_backend_used : "none",
        device_direct_eligible ? 1 : 0,
        device_direct_reason ? device_direct_reason : "not_requested",
        device_direct_live_snapshot,
        device_direct_max_total_bytes,
        device_direct_budget_remaining,
        cpu_access_risk ? cpu_access_risk : "unknown",
        hot_gap_match ? 1 : 0
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
                                        const char* policy_source,
                                        bool gap_watch_class_match,
                                        const char* size_bucket,
                                        cudaStream_t stream,
                                        uintptr_t overlap_start,
                                        uintptr_t overlap_end,
                                        size_t overlap_bytes,
                                        const char* placement_backend,
                                        const char* device_direct_backend_used,
                                        bool device_direct_eligible,
                                        const char* device_direct_reason,
                                        const char* cpu_access_risk,
                                        bool hot_gap_match) {
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
    size_t device_direct_live_snapshot = device_direct_live_bytes.load();
    size_t device_direct_budget_remaining =
        device_direct_budget_remaining_snapshot(device_direct_live_snapshot);

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_ALLOC alloc_id=%zu watch_name=%s "
        "ptr=0x%llx end=0x%llx size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s action=%s policy_source=%s "
        "gap_watch_target_class=%s gap_watch_class_match=%d stream=0x%llx overlap_start=0x%llx "
        "overlap_end=0x%llx overlap_bytes=%zu overlap_ratio_of_watch=%.6f "
        "placement_backend=%s device_direct_backend=%s "
        "device_direct_eligible=%d device_direct_reason=%s "
        "device_direct_live_bytes=%zu device_direct_max_total_bytes=%zu "
        "device_direct_budget_remaining=%zu "
        "cpu_access_risk=%s hot_gap_match=%d\n",
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
        (policy_source && policy_source[0] != '\0') ? policy_source : "unknown",
        gap_watch_target_class.c_str(),
        gap_watch_class_match ? 1 : 0,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),
        static_cast<unsigned long long>(overlap_start),
        static_cast<unsigned long long>(overlap_end),
        overlap_bytes,
        gap_ratio,
        placement_backend ? placement_backend : "managed",
        device_direct_backend_used ? device_direct_backend_used : "none",
        device_direct_eligible ? 1 : 0,
        device_direct_reason ? device_direct_reason : "not_requested",
        device_direct_live_snapshot,
        device_direct_max_total_bytes,
        device_direct_budget_remaining,
        cpu_access_risk ? cpu_access_risk : "unknown",
        hot_gap_match ? 1 : 0
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
        "alloc_policy_source=%s alloc_policy_success=%d alloc_policy_error=%s "
        "gap_watch_target_class=%s gap_watch_policy_action=%s "
        "overlap_start=0x%llx overlap_end=0x%llx overlap_bytes=%zu "
        "overlap_ratio_of_watch=%.6f lifetime_s=%.6f total_bytes=%zu "
        "placement_backend=%s device_direct_backend=%s "
        "device_direct_eligible=%d device_direct_reason=%s "
        "cpu_access_risk=%s hot_gap_match=%d\n",
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
        info->policy_source_name.c_str(),
        info->policy_action_success ? 1 : 0,
        info->policy_action_error.c_str(),
        info->gap_watch_target_class_name.c_str(),
        info->gap_watch_policy_action_name.c_str(),
        static_cast<unsigned long long>(info->gap_overlap_start),
        static_cast<unsigned long long>(info->gap_overlap_end),
        info->gap_overlap_bytes,
        gap_ratio,
        lifetime_seconds,
        current_total,
        info->placement_backend_name.c_str(),
        info->device_direct_backend_name.c_str(),
        info->device_direct_eligible ? 1 : 0,
        info->device_direct_reason.c_str(),
        info->cpu_access_risk.c_str(),
        info->hot_gap_match ? 1 : 0
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
    PolicyAction base_policy_action = choose_policy_action(
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
    bool gap_watch_class_match = false;
    const char* policy_source = "base_policy";
    PolicyAction gap_watch_policy_action = choose_gap_watch_policy_action(
        alloc_class,
        phase_snapshot,
        static_cast<size_t>(size),
        device,
        gap_overlap_bytes,
        &gap_watch_class_match,
        &policy_source
    );
    PolicyAction policy_action =
        (gap_watch_policy_action != PolicyAction::ManagedDefault)
            ? gap_watch_policy_action
            : base_policy_action;
    bool hot_gap_match = gap_watch_enabled && gap_overlap_bytes > 0;
    bool device_direct_requested = is_device_direct_action(policy_action);
    bool device_direct_phase_match = is_device_direct_target_phase(phase_snapshot);
    const char* placement_backend = "managed";
    const char* device_direct_backend_used = "none";
    const char* cpu_access_risk = device_direct_phase_match
        ? "low_no_cpu_access_evidence_runtime_phase"
        : "unknown";
    const char* device_direct_reason = "not_requested";
    bool device_direct_eligible = false;

    if (device_direct_requested) {
        if (device < 0) {
            device_direct_reason = "invalid_device";
        } else if (!hot_gap_match) {
            device_direct_reason = "no_hot_gap_match";
        } else if (!gap_watch_class_match) {
            device_direct_reason = "target_class_mismatch";
        } else if (!device_direct_phase_match) {
            device_direct_reason = "phase_not_allowed";
        } else if (static_cast<size_t>(size) < device_direct_min_bytes) {
            device_direct_reason = "below_min_bytes";
        } else if (static_cast<size_t>(size) > device_direct_max_bytes) {
            device_direct_reason = "above_max_bytes";
        } else {
            device_direct_eligible = true;
            if (policy_action == PolicyAction::DeviceDirect && device_direct_enable) {
                device_direct_reason = "device_direct_enabled";
            } else if (policy_action == PolicyAction::DeviceDirectTrace) {
                device_direct_reason = "trace_action_only";
            } else {
                device_direct_reason = "trace_only_not_enabled";
            }
        }
        device_direct_trace_allocs.fetch_add(1);
        device_direct_requested_bytes.fetch_add(static_cast<size_t>(size));
        if (device_direct_eligible) {
            device_direct_eligible_allocs.fetch_add(1);
        }
    }

    if (device_direct_eligible &&
        policy_action == PolicyAction::DeviceDirect &&
        device_direct_enable &&
        ptr != NULL) {
        void* device_ptr = NULL;
        bool budget_reserved = reserve_device_direct_budget(static_cast<size_t>(size));
        if (!budget_reserved) {
            device_direct_reason = "device_direct_budget_exceeded";
            device_direct_budget_rejects.fetch_add(1);
        } else {
            int previous_device = -1;
            cudaError_t get_device_err = cudaGetDevice(&previous_device);
            if (device >= 0) {
                cudaSetDevice(device);
            }
            bool use_async_backend = device_direct_backend == "cuda_malloc_async";
            cudaError_t device_alloc_err = use_async_backend
                ? cudaMallocAsync(&device_ptr, static_cast<size_t>(size), stream)
                : cudaMalloc(&device_ptr, static_cast<size_t>(size));
            if (get_device_err == cudaSuccess && previous_device >= 0) {
                cudaSetDevice(previous_device);
            }

            if (device_alloc_err == cudaSuccess && device_ptr != NULL) {
                cudaError_t free_candidate_err = cudaFree(ptr);
                if (free_candidate_err == cudaSuccess) {
                    ptr = device_ptr;
                    placement_backend = "device_direct";
                    device_direct_backend_used = device_direct_backend.c_str();
                    policy_action_success = true;
                    policy_action_error = "none";
                    device_direct_reason = "device_direct_enabled";
                    device_direct_actual_allocs.fetch_add(1);
                    device_direct_actual_bytes.fetch_add(static_cast<size_t>(size));
                } else {
                    cudaError_t cleanup_device_err = use_async_backend
                        ? cudaFreeAsync(device_ptr, stream)
                        : cudaFree(device_ptr);
                    if (cleanup_device_err == cudaSuccess) {
                        release_device_direct_budget(static_cast<size_t>(size));
                    }
                    policy_action_success = false;
                    policy_action_error = cudaGetErrorString(free_candidate_err);
                    device_direct_reason = "managed_candidate_free_failed";
                    device_direct_fallback_allocs.fetch_add(1);
                }
            } else {
                release_device_direct_budget(static_cast<size_t>(size));
                policy_action_success = false;
                policy_action_error = cudaGetErrorString(device_alloc_err);
                device_direct_reason = use_async_backend
                    ? "device_malloc_async_failed_fallback_managed"
                    : "device_malloc_failed_fallback_managed";
                device_direct_fallback_allocs.fetch_add(1);
            }
        }
    }

    // 结构化的详细追踪日志，用于后续的离线分析或 eBPF 性能重放关联
    if (size >= trace_min_bytes) {
        trace_allocation_event(
            ptr, static_cast<size_t>(size), alloc_count, current,
            peak_allocated.load(), device, phase_snapshot
        );
    }

    if (gap_overlap_bytes > 0) {
        gap_watch_overlap_allocs.fetch_add(1);
        gap_watch_overlap_bytes_total.fetch_add(gap_overlap_bytes);
        if (gap_watch_class_match) {
            gap_watch_target_class_match_allocs.fetch_add(1);
        }
    }
    bool store_active_info =
        static_cast<size_t>(size) >= trace_min_bytes ||
        log_unknown_detail || log_gap_watch ||
        strcmp(placement_backend, "device_direct") == 0;

    // 7.1 记录活跃分配元数据，供 free / gap watch / unknown detail 使用
    if (store_active_info) {
        std::lock_guard<std::mutex> lock(log_mutex);
        active_allocations[ptr] = AllocationInfo{
            static_cast<size_t>(size),
            device,
            alloc_count,
            phase_snapshot,
            alloc_elapsed,
            size_bucket,
            allocation_class_to_string(alloc_class),
            policy_action_to_string(policy_action),
            policy_source,
            true,
            "none",
            log_unknown_detail,
            log_gap_watch,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes,
            gap_watch_target_class,
            policy_action_to_string(gap_watch_policy_action),
            hot_gap_match,
            placement_backend,
            device_direct_backend_used,
            device_direct_eligible,
            device_direct_reason,
            cpu_access_risk,
        };
    }

    // 8. 策略执行阶段：主动显存调优
    if ((policy_action == PolicyAction::ManagedPrefetchGpu ||
         policy_action == PolicyAction::ManagedAdvisePrefetchGpu) &&
        ptr != NULL) {
        
        // 8.1 内存建议 (MemAdvise)：提示 CUDA 驱动该内存的首选物理位置在 GPU 上
        // 这可以在真正发生缺页中断时，降低驱动寻找最佳页存放地的开销。
        bool should_advise =
            policy_action == PolicyAction::ManagedAdvisePrefetchGpu ||
            policy_warmup_advise_gpu;
        if (should_advise) {
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

    if (gap_overlap_bytes > 0 &&
        policy_source != nullptr &&
        strcmp(policy_source, "gap_watch_policy") == 0 &&
        policy_action != PolicyAction::ManagedDefault) {
        gap_watch_policy_applied_allocs.fetch_add(1);
        gap_watch_policy_applied_overlap_bytes.fetch_add(gap_overlap_bytes);
        if (policy_action_success) {
            gap_watch_policy_success_allocs.fetch_add(1);
        } else {
            gap_watch_policy_failed_allocs.fetch_add(1);
        }
    }

    if (store_active_info) {
        std::lock_guard<std::mutex> lock(log_mutex);
        auto it = active_allocations.find(ptr);
        if (it != active_allocations.end()) {
            it->second.policy_action_name = policy_action_to_string(policy_action);
            it->second.policy_source_name = policy_source ? policy_source : "unknown";
            it->second.policy_action_success = policy_action_success;
            it->second.policy_action_error = policy_action_error;
            it->second.gap_watch_target_class_name = gap_watch_target_class;
            it->second.gap_watch_policy_action_name =
                policy_action_to_string(gap_watch_policy_action);
            it->second.placement_backend_name = placement_backend;
            it->second.device_direct_backend_name = device_direct_backend_used;
            it->second.device_direct_eligible = device_direct_eligible;
            it->second.device_direct_reason = device_direct_reason;
            it->second.cpu_access_risk = cpu_access_risk;
            it->second.hot_gap_match = hot_gap_match;
        }
    }

    // 记录策略引擎的判定与执行结果
    if (policy_enabled) {
        trace_policy_event(
            ptr, static_cast<size_t>(size), alloc_count, device,
            phase_snapshot, alloc_class, policy_action,
            policy_source,
            gap_watch_class_match,
            gap_overlap_bytes,
            size_bucket,
            placement_backend,
            device_direct_backend_used,
            device_direct_eligible,
            device_direct_reason,
            cpu_access_risk,
            hot_gap_match,
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
            policy_source,
            gap_watch_class_match,
            size_bucket,
            stream,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes,
            placement_backend,
            device_direct_backend_used,
            device_direct_eligible,
            device_direct_reason,
            cpu_access_risk,
            hot_gap_match
        );
    }

    // 9. 全局暴力预取 Fallback (通常默认关闭)
    // 如果系统开启了全局预取 (enable_prefetch=1)，且策略引擎没有覆盖该内存块，则执行普适性预取。
    // 在生产环境的推理服务中，由于 PCIe 带宽的脆弱性，此选项通常不建议保持常开。
    if (enable_prefetch && device >= 0 && ptr != NULL &&
        policy_action != PolicyAction::ManagedPrefetchGpu &&
        policy_action != PolicyAction::ManagedAdvisePrefetchGpu) {
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

        bool is_device_direct =
            has_info && info.placement_backend_name == "device_direct";
        bool use_async_backend =
            is_device_direct &&
            info.device_direct_backend_name == "cuda_malloc_async";
        cudaError_t err = use_async_backend
            ? cudaFreeAsync(ptr, stream)
            : cudaFree(ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[vLLM UVM] %s failed: %s\n",
                    use_async_backend ? "cudaFreeAsync" : "cudaFree",
                    cudaGetErrorString(err));
        }

        // Update statistics
        size_t current = total_allocated.fetch_sub(size) - size;
        size_t free_count = num_frees.fetch_add(1) + 1;

        if (has_info &&
            info.placement_backend_name == "device_direct" &&
            err == cudaSuccess) {
            release_device_direct_budget(info.size);
            device_direct_free_success_allocs.fetch_add(1);
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
    device_direct_live_bytes.store(0);
    device_direct_peak_live_bytes.store(0);
    device_direct_budget_rejects.store(0);
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
    fprintf(log_file, "  Gap-watch overlap allocations: %zu\n", gap_watch_overlap_allocs.load());
    fprintf(log_file, "  Gap-watch overlap bytes total: %zu\n", gap_watch_overlap_bytes_total.load());
    fprintf(log_file, "  Gap-watch target-class matches: %zu\n", gap_watch_target_class_match_allocs.load());
    fprintf(log_file, "  Gap-watch policy applied: %zu\n", gap_watch_policy_applied_allocs.load());
    fprintf(log_file, "  Gap-watch policy applied overlap bytes: %zu\n", gap_watch_policy_applied_overlap_bytes.load());
    fprintf(log_file, "  Gap-watch policy success: %zu\n", gap_watch_policy_success_allocs.load());
    fprintf(log_file, "  Gap-watch policy failed: %zu\n", gap_watch_policy_failed_allocs.load());
    fprintf(log_file, "  Device-direct trace allocations: %zu\n", device_direct_trace_allocs.load());
    fprintf(log_file, "  Device-direct eligible allocations: %zu\n", device_direct_eligible_allocs.load());
    fprintf(log_file, "  Device-direct requested bytes: %zu\n", device_direct_requested_bytes.load());
    fprintf(log_file, "  Device-direct backend: %s\n", device_direct_backend.c_str());
    fprintf(log_file, "  Device-direct max total bytes: %zu\n", device_direct_max_total_bytes);
    fprintf(log_file, "  Device-direct actual allocations: %zu\n", device_direct_actual_allocs.load());
    fprintf(log_file, "  Device-direct actual bytes: %zu\n", device_direct_actual_bytes.load());
    fprintf(log_file, "  Device-direct live bytes: %zu\n", device_direct_live_bytes.load());
    fprintf(log_file, "  Device-direct peak live bytes: %zu\n", device_direct_peak_live_bytes.load());
    fprintf(log_file, "  Device-direct budget rejects: %zu\n", device_direct_budget_rejects.load());
    fprintf(log_file, "  Device-direct fallback allocations: %zu\n", device_direct_fallback_allocs.load());
    fprintf(log_file, "  Device-direct free success: %zu\n", device_direct_free_success_allocs.load());
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
