#!/usr/bin/env bash
# Single-process KV fault ratio runner for vLLM + UVM.
#
# Flow (same server process):
# 1) start ONE vLLM server process
# 2) wait until a fresh kv_cache range appears in vLLM address log
# 3) configure nvidia_uvm KV range params immediately
# 4) run benchmark against that SAME server process
# 5) print fault stats and cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"

MODEL="Qwen/Qwen3-30B-A3B-FP8"
PORT=8000
PROMPTS=20
REQUEST_RATE=5
OUTPUT_LEN=512
SEED=42
STARTUP_TIMEOUT=600
CHECK_INTERVAL=2

DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
ADDRESS_LOG="/tmp/vllm_uvm_address_regions.log"
SERVER_LOG="$SCRIPT_DIR/vllm_server_uvm_single_process.log"
BENCH_LOG="$SCRIPT_DIR/results/uvm_kv_bench_single_process.log"
TRACE_LOG=""  # auto-generated in validate_inputs if not set via --trace-log
ADDRESS_TRACE_LOG=""  # auto-generated when --with-address-log and not set via --address-trace-log
ALLOCATOR_TRACE_LOG=""  # optional UVM allocator trace log
MODE="dmesg" # dmesg | trace
TRACE_DIR=""
UVM_TRACE_MIN_BYTES="${UVM_TRACE_MIN_BYTES:-1048576}"

ENABLE_FAULT_ADDRESS_LOG=0
KEEP_ON_EXIT=0
NO_CLEANUP=0
NO_BENCH=0
RESET_COUNTERS=1
UVM_KO_PATH="$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
UVM_ENABLE_DEBUG_PROCFS=1

KV_START=""
KV_END=""
SERVER_PID=""
TRACE_PID=""
DMESG_PID=""
RUN_STATS_LOG=""
RUN_ADDRESS_LOG=""
TRACE_READER_ERROR_LOG=""

BENCH_CMD=()

# Stats line markers:
# - legacy English: "Replayable fault stats GPU" + key=value fields
# - newer format may include Chinese labels (e.g. "本批次总缺页实例数=")
STATS_LINE_PATTERN='Replayable fault stats GPU|batch_faults=|本批次总缺页实例数='

usage() {
  cat <<'USAGE'
Usage:
  run_kv_fault_ratio.sh [options] [-- <custom bench command...>]

Options:
  --model <name>             vLLM model (default: Qwen/Qwen3-30B-A3B-FP8)
  --port <int>               vLLM server port (default: 8000)
  --prompts <int>            Benchmark prompts (default: 50)
  --request-rate <float>     Benchmark request-rate (default: 5)
  --output-len <int>         sharegpt output len (default: 512)
  --seed <int>               Benchmark seed (default: 42)
  --dataset-path <path>      ShareGPT dataset path
  --address-log <path>       vLLM address log path
  --server-log <path>        Server log file path
  --bench-log <path>         Benchmark output log path
  --mode <dmesg|trace>       Fault stat destination mode (default: dmesg)
  --trace-log <path>         Replayable fault stats capture file
  --address-trace-log <path> Per-fault address capture file (implies --with-address-log)
  --allocator-log <path>     UVM allocator trace file for alloc/free/phase events
  --uvm-trace-min-bytes <n>  Min allocation size to record in allocator trace (default: 1048576)
  --startup-timeout <sec>    Server/KV wait timeout (default: 600)
  --check-interval <sec>     Poll interval (default: 2)
  --with-address-log         Also enable per-fault address logs
  --no-cleanup               Skip workloads/cleanup_gpu.py before run
  --no-bench                 Configure params only, do not run benchmark
  --reset-counters           Reset total_* counters by reloading nvidia_uvm (default: on)
  --no-reset-counters        Do not reload nvidia_uvm before run
  --uvm-ko-path <path>       nvidia-uvm.ko path for --reset-counters
  --uvm-enable-debug-procfs <0|1>  Set at insmod time (default: 1)
  --keep                     Keep UVM params on exit (no rollback)
  -h, --help                 Show help

Custom bench command:
  If provided after '--', it runs against the same server process after KV params are configured.

Examples:
  # One-click same-process run (recommended)
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --trace-log /tmp/uvm_kv_fault_trace.log

  # Split stats and per-fault address logs into two files
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --with-address-log \
    --trace-log /tmp/uvm_kv_fault_stats.log \
    --address-trace-log /tmp/uvm_kv_fault_addrs.log

  # Split stats/address logs and also capture allocator trace
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --with-address-log \
    --trace-log /tmp/uvm_kv_fault_stats.log \
    --address-trace-log /tmp/uvm_kv_fault_addrs.log \
    --allocator-log /tmp/vllm_uvm_allocator_trace.log \
    --uvm-trace-min-bytes 1048576

  # Custom bench command
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace -- \
    uv run vllm bench serve --model Qwen/Qwen3-30B-A3B-FP8 \
      --dataset-name sharegpt --dataset-path workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 100 --sharegpt-output-len 512 --seed 42 --request-rate 5 --port 8000
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_file() {
  local path="$1"
  [ -e "$path" ] || die "Missing required path: $path"
}

need_writable_param() {
  local key="$1"
  sudo test -w "$PARAM_DIR/$key" || die "Parameter is not writable: $PARAM_DIR/$key"
}

write_param() {
  local key="$1"
  local value="$2"
  echo "$value" | sudo tee "$PARAM_DIR/$key" >/dev/null
}

is_hex_u64() {
  [[ "$1" =~ ^0x[0-9a-fA-F]+$ ]]
}

extract_kv_range_from_log() {
  local line
  line="$(grep -E 'kv_cache:(contiguous_range|span_range),all_layers,0x[0-9a-fA-F]+,0x[0-9a-fA-F]+' "$ADDRESS_LOG" | tail -n 1 || true)"
  [ -n "$line" ] || return 1

  local kind name start end
  IFS=',' read -r kind name start end _ <<<"$line"
  is_hex_u64 "$start" || return 1
  is_hex_u64 "$end" || return 1

  KV_START="$start"
  KV_END="$end"
  return 0
}

# 启动 vLLM 服务器进程
start_server() {
    local allocator_env=""

    # 1. 环境准备：创建日志文件夹
    # dirname 获取文件的父目录，mkdir -p 确保路径存在且不重复报错
    mkdir -p "$(dirname "$SERVER_LOG")"
    mkdir -p "$(dirname "$ADDRESS_LOG")"
    if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
      mkdir -p "$(dirname "$ALLOCATOR_TRACE_LOG")"
    fi

    # 2. 初始化日志文件
    # ': >' 是 Bash 的一种高效清空文件内容的方法（不启动外部进程）
    # 确保每次运行压测时，之前的旧日志不会干扰本次统计
    : > "$SERVER_LOG"
    : > "$ADDRESS_LOG"
    if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
      : > "$ALLOCATOR_TRACE_LOG"
      allocator_env="VLLM_UVM_LOG_FILE='$ALLOCATOR_TRACE_LOG' VLLM_UVM_TRACE_MIN_BYTES='$UVM_TRACE_MIN_BYTES'"
    fi

    # 3. 构造服务器启动指令
    local server_cmd
    # 指令解析：
    # - cd '$SCRIPT_DIR': 切换到脚本目录，确保相对路径正确
    # - VLLM_USE_UVM=1: 强制 vLLM 使用统一虚拟内存 (UVM) 模式
    # - VLLM_UVM_ADDRESS_LOG_ENABLE=1: 开启 vLLM 的内存地址探测插件
    # - VLLM_UVM_ADDRESS_LOG_FILE: 指定 vLLM 将内存布局信息写到哪个文件
    # - uv run vllm serve: 使用 uv 工具启动 vLLM 推理引擎
    # - --enforce-eager: 强制使用 Eager 模式（通常为了避免 CUDA Graph 干扰显存地址分析）
    # - --max-num-seqs 16: 限制并发序列数，便于观察显存受限时的行为
    # - > '$SERVER_LOG' 2>&1: 将标准输出和错误信息全部重定向到服务器日志
    server_cmd="cd '$SCRIPT_DIR' && \
                VLLM_USE_UVM=1 \
                VLLM_UVM_ADDRESS_LOG_ENABLE=1 \
                VLLM_UVM_ADDRESS_LOG_FILE='$ADDRESS_LOG' \
                $allocator_env \
                uv run vllm serve '$MODEL' \
                --enforce-eager \
                --max-num-seqs 16 \
                --port '$PORT' > '$SERVER_LOG' 2>&1"

    # 4. 后台执行并脱离终端控制
    # - setsid: 创建一个新的会话（Session）。
    #   这样做的好处是即便当前脚本所在的终端关闭，vLLM 进程组也不会被系统杀死。
    # - bash -lc: 使用 login shell 模式执行命令，确保加载了用户环境（如 PATH, CUDA 路径等）。
    # - &: 将整个进程放入后台运行。
    setsid bash -lc "$server_cmd" &
    
    # 5. 获取并检查进程状态
    # $! 是 Bash 中最近一个后台运行进程的 PID (进程 ID)
    SERVER_PID=$!
    
    # 给服务器一点点启动时间（实例化 Python 解释器的时间）
    sleep 1
    
    # 6. 进程存活检查
    # kill -0 并不发送杀死信号，而是检查该 PID 是否存在且用户是否有权限操作。
    # 如果进程启动失败（例如显存不足或端口冲突），则调用 die 函数终止整个脚本。
    kill -0 "$SERVER_PID" 2>/dev/null || die "Failed to start vLLM server process"
}

stop_server_if_needed() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # Kill whole process group to avoid child leftovers
    kill -TERM -- "-$SERVER_PID" >/dev/null 2>&1 || true
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL -- "-$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

wait_for_server_and_kv_range() {
  local start_ts now elapsed
  start_ts="$(date +%s)"

  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "===== server log tail =====" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      die "Server exited before ready"
    fi

    if extract_kv_range_from_log; then
      if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        return 0
      fi
    fi

    now="$(date +%s)"
    elapsed=$(( now - start_ts ))
    if [ "$elapsed" -ge "$STARTUP_TIMEOUT" ]; then
      echo "===== server log tail =====" >&2
      tail -n 120 "$SERVER_LOG" >&2 || true
      echo "===== address log tail =====" >&2
      tail -n 60 "$ADDRESS_LOG" >&2 || true
      die "Timeout waiting for server readiness and KV range (>${STARTUP_TIMEOUT}s)"
    fi

    sleep "$CHECK_INTERVAL"
  done
}

start_trace_capture_if_needed() {
  [ "$MODE" = "trace" ] || return 0

  resolve_trace_dir
  need_file "$TRACE_DIR/trace_pipe"

  mkdir -p "$(dirname "$TRACE_LOG")"
  RUN_STATS_LOG="$TRACE_LOG"
  : > "$RUN_STATS_LOG"

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    mkdir -p "$(dirname "$ADDRESS_TRACE_LOG")"
    RUN_ADDRESS_LOG="$ADDRESS_TRACE_LOG"
    : > "$RUN_ADDRESS_LOG"
  else
    RUN_ADDRESS_LOG=""
  fi
  TRACE_READER_ERROR_LOG="/tmp/uvm_trace_reader_$$.log"
  : > "$TRACE_READER_ERROR_LOG"

  sudo sh -c ": > '$TRACE_DIR/trace'"
  echo 1 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null

  if sudo fuser "$TRACE_DIR/trace_pipe" >/dev/null 2>&1; then
    echo 0 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null || true
    die "trace_pipe is already busy. Another reader is consuming $TRACE_DIR/trace_pipe. Stop that reader first, or rerun with --mode dmesg."
  fi

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    sudo cat "$TRACE_DIR/trace_pipe" | awk -v stats="$RUN_STATS_LOG" -v addrs="$RUN_ADDRESS_LOG" '
      /Replayable fault stats GPU|batch_faults=|本批次总缺页实例数=/ { print >> stats; fflush(stats); next }
      /Replayable fault GPU|精确原始地址=/ { print >> addrs; fflush(addrs); next }
    ' 2>"$TRACE_READER_ERROR_LOG" &
  else
    sudo cat "$TRACE_DIR/trace_pipe" 2>"$TRACE_READER_ERROR_LOG" | grep -E --line-buffered "$STATS_LINE_PATTERN" > "$RUN_STATS_LOG" &
  fi

  TRACE_PID=$!
  sleep 1
  if ! kill -0 "$TRACE_PID" 2>/dev/null; then
    echo 0 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null || true
    if [ -s "$TRACE_READER_ERROR_LOG" ]; then
      echo "===== trace reader stderr =====" >&2
      tail -n 20 "$TRACE_READER_ERROR_LOG" >&2 || true
    fi
    die "Failed to start trace capture"
  fi
}

start_dmesg_capture_if_needed() {
  [ "$MODE" = "dmesg" ] || return 0

  local dmesg_follow_arg="--follow"
  if dmesg --help 2>&1 | grep -q -- "--follow-new"; then
    dmesg_follow_arg="--follow-new"
  fi

  mkdir -p "$(dirname "$TRACE_LOG")"
  RUN_STATS_LOG="$TRACE_LOG"
  : > "$RUN_STATS_LOG"

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    mkdir -p "$(dirname "$ADDRESS_TRACE_LOG")"
    RUN_ADDRESS_LOG="$ADDRESS_TRACE_LOG"
    : > "$RUN_ADDRESS_LOG"
  else
    RUN_ADDRESS_LOG=""
  fi

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    sudo dmesg "$dmesg_follow_arg" --time-format iso | awk -v stats="$RUN_STATS_LOG" -v addrs="$RUN_ADDRESS_LOG" '
      /Replayable fault stats GPU|batch_faults=|本批次总缺页实例数=/ { print >> stats; fflush(stats); next }
      /Replayable fault GPU|精确原始地址=/ { print >> addrs; fflush(addrs); next }
    ' &
  else
    sudo dmesg "$dmesg_follow_arg" --time-format iso | grep -E --line-buffered "$STATS_LINE_PATTERN" > "$RUN_STATS_LOG" &
  fi

  DMESG_PID=$!
  sleep 1
  kill -0 "$DMESG_PID" 2>/dev/null || die "Failed to start dmesg capture"
}

reload_uvm_module_for_counter_reset() {
  need_file "$UVM_KO_PATH"
  [ "$UVM_ENABLE_DEBUG_PROCFS" = "0" ] || [ "$UVM_ENABLE_DEBUG_PROCFS" = "1" ] \
    || die "--uvm-enable-debug-procfs must be 0 or 1"

  echo "Resetting UVM total counters by reloading nvidia_uvm..."

  if lsmod | grep -q '^nvidia_uvm'; then
    sudo rmmod nvidia_uvm || die "Failed to rmmod nvidia_uvm (module may still be in use)"
  fi

  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs="$UVM_ENABLE_DEBUG_PROCFS" \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    uvm_perf_fault_kv_range_enable=0 \
    || die "Failed to insmod $UVM_KO_PATH"

  if [ -r "$PARAM_DIR/uvm_enable_debug_procfs" ]; then
    local debug_val
    debug_val="$(cat "$PARAM_DIR/uvm_enable_debug_procfs" 2>/dev/null || true)"
    [ "$debug_val" = "$UVM_ENABLE_DEBUG_PROCFS" ] \
      || die "uvm_enable_debug_procfs expected $UVM_ENABLE_DEBUG_PROCFS, got $debug_val"
  fi

  # Optional: clear old kernel ring lines so dmesg mode only shows this run.
  if [ "$MODE" = "dmesg" ]; then
    sudo dmesg -C >/dev/null 2>&1 || true
  fi
}

stop_trace_capture_if_needed() {
  [ "$MODE" = "trace" ] || return 0

  if [ -n "$TRACE_PID" ] && kill -0 "$TRACE_PID" 2>/dev/null; then
    kill "$TRACE_PID" >/dev/null 2>&1 || true
    # Avoid hanging forever on wait if child teardown is abnormal.
    for _ in $(seq 1 20); do
      kill -0 "$TRACE_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$TRACE_PID" >/dev/null 2>&1 || true
    wait "$TRACE_PID" >/dev/null 2>&1 || true
  fi
  TRACE_PID=""
  if [ -n "$TRACE_DIR" ] && [ -e "$TRACE_DIR/tracing_on" ]; then
    echo 0 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null || true
  fi
  if [ -n "$TRACE_READER_ERROR_LOG" ] && [ -e "$TRACE_READER_ERROR_LOG" ]; then
    rm -f "$TRACE_READER_ERROR_LOG" >/dev/null 2>&1 || true
  fi
  TRACE_READER_ERROR_LOG=""
}

stop_dmesg_capture_if_needed() {
  [ "$MODE" = "dmesg" ] || return 0

  if [ -n "$DMESG_PID" ] && kill -0 "$DMESG_PID" 2>/dev/null; then
    kill "$DMESG_PID" >/dev/null 2>&1 || true
    # Avoid hanging forever on wait if child teardown is abnormal.
    for _ in $(seq 1 20); do
      kill -0 "$DMESG_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$DMESG_PID" >/dev/null 2>&1 || true
    wait "$DMESG_PID" >/dev/null 2>&1 || true
  fi
  DMESG_PID=""
}

resolve_trace_dir() {
  if [ -n "$TRACE_DIR" ] && [ -e "$TRACE_DIR/trace_pipe" ]; then
    return 0
  fi

  # Prefer tracefs mountpoint on modern kernels.
  if [ -e /sys/kernel/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/tracing"
    return 0
  fi

  # Fallback to debugfs-mounted tracing path.
  if [ -e /sys/kernel/debug/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/debug/tracing"
    return 0
  fi

  # Try mounting tracefs first.
  sudo mount -t tracefs nodev /sys/kernel/tracing >/dev/null 2>&1 || true
  if [ -e /sys/kernel/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/tracing"
    return 0
  fi

  # Last fallback: mount debugfs then use tracing under it.
  sudo mount -t debugfs none /sys/kernel/debug >/dev/null 2>&1 || true
  if [ -e /sys/kernel/debug/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/debug/tracing"
    return 0
  fi

  die "Unable to locate trace_pipe in /sys/kernel/tracing or /sys/kernel/debug/tracing"
}

configure_uvm_params() {
  local destination=0
  [ "$MODE" = "trace" ] && destination=1

  write_param uvm_perf_fault_kv_start "$KV_START"
  write_param uvm_perf_fault_kv_end "$KV_END"
  write_param uvm_perf_fault_kv_range_enable 1
  write_param uvm_perf_fault_log_destination "$destination"
  write_param uvm_perf_fault_log_counters 1
  write_param uvm_perf_fault_log_addresses "$ENABLE_FAULT_ADDRESS_LOG"
}

cleanup() {
  stop_dmesg_capture_if_needed || true
  stop_trace_capture_if_needed || true

  if [ "$KEEP_ON_EXIT" -eq 0 ]; then
    write_param uvm_perf_fault_log_counters 0 || true
    write_param uvm_perf_fault_kv_range_enable 0 || true
    write_param uvm_perf_fault_log_addresses 0 || true
    write_param uvm_perf_fault_log_destination 0 || true
  fi

  stop_server_if_needed || true
}

# 解析命令行传入的参数
parse_args() {
  # $# 代表当前传入参数的总个数。
  # 当参数个数大于 0 时，持续循环处理。每次处理完后会通过 shift 踢除已处理的参数。
  while [ $# -gt 0 ]; do
    # $1 代表当前参数列表中的第一个参数（即当前正在检查的选项名）
    case "$1" in
      # ==========================================
      # 键值对类型参数 (Key-Value): 选项后面跟一个具体的值
      # 处理方式: 将 $2 (紧跟在选项后的值) 赋给对应变量，
      # 然后使用 'shift 2' 将这两个参数从参数列表中移除，让下一个选项成为新的 $1。
      # ==========================================
      --model) MODEL="$2"; shift 2 ;;                       # 设置模型名称
      --port) PORT="$2"; shift 2 ;;                         # 设置服务端口
      --prompts) PROMPTS="$2"; shift 2 ;;                   # 设置测试使用的 prompt 数量
      --request-rate) REQUEST_RATE="$2"; shift 2 ;;         # 设置请求发送速率 (QPS)
      --output-len) OUTPUT_LEN="$2"; shift 2 ;;             # 设置生成的输出 token 长度
      --seed) SEED="$2"; shift 2 ;;                         # 设置随机种子
      --dataset-path) DATASET_PATH="$2"; shift 2 ;;         # 设置数据集的文件路径
      --address-log) ADDRESS_LOG="$2"; shift 2 ;;           # 设置 vLLM 地址日志的输出路径
      --server-log) SERVER_LOG="$2"; shift 2 ;;             # 设置服务端运行日志的路径
      --bench-log) BENCH_LOG="$2"; shift 2 ;;               # 设置压测结果日志的路径
      --mode) MODE="$2"; shift 2 ;;                         # 设置统计模式 (dmesg 还是 trace)
      --trace-log) TRACE_LOG="$2"; shift 2 ;;               # 设置 trace 日志的输出路径
      --address-trace-log) ADDRESS_TRACE_LOG="$2"; ENABLE_FAULT_ADDRESS_LOG=1; shift 2 ;; # 设置地址日志路径（并自动开启地址记录）
      --allocator-log) ALLOCATOR_TRACE_LOG="$2"; shift 2 ;; # 设置 allocator trace 日志路径
      --uvm-trace-min-bytes) UVM_TRACE_MIN_BYTES="$2"; shift 2 ;; # 设置 allocator trace 最小阈值
      --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;   # 设置等待服务器启动的超时时间
      --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;     # 设置轮询检查的间隔时间

      # ==========================================
      # 布尔开关类型参数 (Boolean Flags): 只需要出现该选项即可，不需要值
      # 处理方式: 将对应的标志变量设为 1 (True)，
      # 然后使用 'shift' (等同于 shift 1) 仅移除当前这个选项本身。
      # ==========================================
      --with-address-log) ENABLE_FAULT_ADDRESS_LOG=1; shift ;; # 开启具体缺页地址记录
      --no-cleanup) NO_CLEANUP=1; shift ;;                     # 禁用运行前的环境清理
      --no-bench) NO_BENCH=1; shift ;;                         # 仅配置环境，不实际执行压测
      --reset-counters) RESET_COUNTERS=1; shift ;;             # 运行前重载 nvidia_uvm 清零累计计数
      --no-reset-counters) RESET_COUNTERS=0; shift ;;          # 不重载模块，保留累计计数
      --uvm-ko-path) UVM_KO_PATH="$2"; shift 2 ;;              # 指定 nvidia-uvm.ko 的绝对路径
      --uvm-enable-debug-procfs) UVM_ENABLE_DEBUG_PROCFS="$2"; shift 2 ;; # insmod 时的 debug procfs 开关
      --keep) KEEP_ON_EXIT=1; shift ;;                         # 退出时不重置内核状态

      # ==========================================
      # 帮助与特殊标识
      # ==========================================
      -h|--help) 
        usage;    # 调用外部定义的 usage 函数打印帮助信息
        exit 0 ;; # 正常退出脚本
      
      --) 
        # '--' 是 Linux 命令行的标准约定，表示“选项解析到此结束”。
        # 后面出现的所有内容，即使以 '-' 开头，也会被当作普通参数/命令对待。
        shift # 移除 '--' 这个符号本身
        
        # "$@" 代表当前列表中剩下的所有参数。
        # 将它们打包存入 BENCH_CMD 数组中，作为用户自定义的实际压测命令。
        BENCH_CMD=("$@") 
        
        break ;; # 结束 while 循环，停止解析后续参数
        
      *) 
        # 默认匹配 (类似 switch-case 的 default)
        # 如果输入了脚本不认识的选项，调用 die 函数报错并退出。
        die "Unknown argument: $1 (use --help)" ;; 
    esac
  done
}

# 验证输入参数、检查环境依赖并构建默认压测命令
validate_inputs() {
    # 1. 验证统计模式：必须是 dmesg 或 trace 其中之一
    # [ A ] || [ B ] || die 表示：如果 A 不成立且 B 也不成立，则触发 die 函数报错
    [ "$MODE" = "dmesg" ] || [ "$MODE" = "trace" ] || die "--mode must be dmesg or trace"

    # 2. 检查系统工具依赖：确保系统已安装 curl（用于检查 vLLM 服务器是否存活）
    # command -v 用于寻找命令的路径，如果找不到会返回非零状态码
    command -v curl >/dev/null 2>&1 || die "curl is required but not found"

    # 3. 验证 NVIDIA UVM 内核参数接口：
    # 依次检查定义的 sysfs 文件是否存在。如果这些文件缺失，说明 nvidia_uvm 驱动版本不匹配
    # 或者没有打上支持 KV Cache 统计的内核补丁。
    need_file "$PARAM_DIR/uvm_perf_fault_log_counters"
    need_file "$PARAM_DIR/uvm_perf_fault_log_destination"
  need_file "$PARAM_DIR/uvm_perf_fault_log_addresses"
  need_file "$PARAM_DIR/uvm_perf_fault_kv_range_enable"
  need_file "$PARAM_DIR/uvm_perf_fault_kv_start"
  need_file "$PARAM_DIR/uvm_perf_fault_kv_end"
  need_writable_param "uvm_perf_fault_log_counters"
  need_writable_param "uvm_perf_fault_log_destination"
  need_writable_param "uvm_perf_fault_log_addresses"
  need_writable_param "uvm_perf_fault_kv_range_enable"
  need_writable_param "uvm_perf_fault_kv_start"
  need_writable_param "uvm_perf_fault_kv_end"

    # 4. 验证数据集文件：确保指定的 ShareGPT 等测试数据集路径有效
    need_file "$DATASET_PATH"

    # 4.1 验证 allocator trace 阈值
    [[ "$UVM_TRACE_MIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-trace-min-bytes must be a non-negative integer"

    # 5. 日志目录初始化：创建压测日志所在的文件夹
    mkdir -p "$(dirname "$BENCH_LOG")"

    # 6. 自动生成 stats/address 日志文件名（若用户未显式指定）
    if [ -z "$TRACE_LOG" ]; then
      local ts
      ts="$(date +%Y%m%d_%H%M%S)"
      TRACE_LOG="/tmp/uvm_kv_fault_stats_${ts}.log"
      if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] && [ -z "$ADDRESS_TRACE_LOG" ]; then
        ADDRESS_TRACE_LOG="/tmp/uvm_kv_fault_addrs_${ts}.log"
      fi
    elif [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] && [ -z "$ADDRESS_TRACE_LOG" ]; then
      local ts
      ts="$(date +%Y%m%d_%H%M%S)"
      ADDRESS_TRACE_LOG="/tmp/uvm_kv_fault_addrs_${ts}.log"
    fi

    # 如果用户在命令行没有通过 '--' 传入自定义命令 (${#BENCH_CMD[@]} -eq 0)
    # 且用户没有开启 '--no-bench' (表示需要运行压测)，则初始化默认的 vllm 压测指令
    if [ ${#BENCH_CMD[@]} -eq 0 ] && [ "$NO_BENCH" -eq 0 ]; then
        # 构建一个数组，存储 vllm bench serve 的完整参数
        BENCH_CMD=(
            uv run --directory "$SCRIPT_DIR" vllm bench serve
            --model "$MODEL"                # 使用的模型名称
            --dataset-name sharegpt         # 数据集格式
            --dataset-path "$DATASET_PATH"  # 数据集具体路径
            --num-prompts "$PROMPTS"        # 发送的请求总数
            --sharegpt-output-len "$OUTPUT_LEN" # 限制输出长度
            --seed "$SEED"                  # 随机种子
            --request-rate "$REQUEST_RATE"  # 请求频率 (QPS)
            --port "$PORT"                  # vLLM 服务器端口
        )
    fi
}

print_recent_stats() {
  echo "===== Recent Replayable fault stats ====="
  grep -E "$STATS_LINE_PATTERN" "$RUN_STATS_LOG" | tail -n 20 2>/dev/null || true
}

extract_stat_int() {
  local line="$1"
  local key="$2"

  # Preferred parse path: machine-readable key=value fields.
  local value
  value="$(echo "$line" | sed -n "s/.*${key}=\([0-9][0-9]*\).*/\1/p")"
  if [ -n "$value" ]; then
    echo "$value"
    return 0
  fi

  # Partial fallback for localized stats lines.
  # NOTE: Without key=value fields some counters can be ambiguous/unavailable.
  case "$key" in
    batch_faults)
      echo "$line" | sed -n 's/.*本批次总缺页实例数=\([0-9][0-9]*\).*/\1/p'
      ;;
    batch_after_dedup)
      echo "$line" | sed -n 's/.*本批次总缺页实例数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    batch_kv_faults)
      echo "$line" | sed -n 's/.*KV类的总缺页数=\([0-9][0-9]*\).*/\1/p'
      ;;
    batch_kv_duplicates)
      echo "$line" | sed -n 's/.*KV类的总缺页数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_faults)
      echo "$line" | sed -n 's/.*|| 总缺页数=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_after_dedup)
      echo "$line" | sed -n 's/.*|| 总缺页数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_kv_faults)
      echo "$line" | sed -n 's/.*kv总错误数=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_kv_after_dedup)
      echo "$line" | sed -n 's/.*kv总错误数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    *)
      true
      ;;
  esac
}

safe_non_negative_delta() {
  local end="$1"
  local begin="$2"
  if [ "$end" -lt "$begin" ]; then
    echo 0
  else
    echo $(( end - begin ))
  fi
}

calc_ratio_percent() {
  local num="$1"
  local den="$2"
  awk -v n="$num" -v d="$den" 'BEGIN { if (d <= 0) printf "0.00"; else printf "%.2f", (100.0 * n) / d }'
}

print_delta_stats() {
  local stats_file="$RUN_STATS_LOG"
  [ -n "$stats_file" ] || return 0
  [ -s "$stats_file" ] || {
    echo "===== Delta Replayable fault stats (this workload) ====="
    echo "No stats captured in $stats_file"
    return 0
  }

  local first_line last_line
  first_line="$(grep -E "$STATS_LINE_PATTERN" "$stats_file" | head -n 1 || true)"
  last_line="$(grep -E "$STATS_LINE_PATTERN" "$stats_file" | tail -n 1 || true)"
  [ -n "$first_line" ] || return 0
  [ -n "$last_line" ] || return 0

  local first_batch_faults first_batch_duplicates first_batch_after_dedup
  local first_batch_kv_faults first_batch_kv_duplicates first_batch_kv_after_dedup
  local first_total_faults first_total_duplicates first_total_after_dedup
  local first_total_kv_faults first_total_kv_duplicates first_total_kv_after_dedup

  local last_total_faults last_total_duplicates last_total_after_dedup
  local last_total_kv_faults last_total_kv_duplicates last_total_kv_after_dedup

  first_batch_faults="$(extract_stat_int "$first_line" "batch_faults")"
  first_batch_duplicates="$(extract_stat_int "$first_line" "batch_duplicates")"
  first_batch_after_dedup="$(extract_stat_int "$first_line" "batch_after_dedup")"
  first_batch_kv_faults="$(extract_stat_int "$first_line" "batch_kv_faults")"
  first_batch_kv_duplicates="$(extract_stat_int "$first_line" "batch_kv_duplicates")"
  first_batch_kv_after_dedup="$(extract_stat_int "$first_line" "batch_kv_after_dedup")"

  first_total_faults="$(extract_stat_int "$first_line" "total_faults")"
  first_total_duplicates="$(extract_stat_int "$first_line" "total_duplicates")"
  first_total_after_dedup="$(extract_stat_int "$first_line" "total_after_dedup")"
  first_total_kv_faults="$(extract_stat_int "$first_line" "total_kv_faults")"
  first_total_kv_duplicates="$(extract_stat_int "$first_line" "total_kv_duplicates")"
  first_total_kv_after_dedup="$(extract_stat_int "$first_line" "total_kv_after_dedup")"

  last_total_faults="$(extract_stat_int "$last_line" "total_faults")"
  last_total_duplicates="$(extract_stat_int "$last_line" "total_duplicates")"
  last_total_after_dedup="$(extract_stat_int "$last_line" "total_after_dedup")"
  last_total_kv_faults="$(extract_stat_int "$last_line" "total_kv_faults")"
  last_total_kv_duplicates="$(extract_stat_int "$last_line" "total_kv_duplicates")"
  last_total_kv_after_dedup="$(extract_stat_int "$last_line" "total_kv_after_dedup")"

  # Derive duplicate counters from fault totals and after-dedup values when the
  # localized stats line does not expose explicit "*_duplicates" fields.
  [ -n "$first_batch_duplicates" ] || first_batch_duplicates=$(( first_batch_faults - first_batch_after_dedup ))
  [ -n "$first_batch_kv_duplicates" ] || first_batch_kv_duplicates=$(( first_batch_kv_faults - first_batch_kv_after_dedup ))
  [ -n "$first_total_duplicates" ] || first_total_duplicates=$(( first_total_faults - first_total_after_dedup ))
  [ -n "$first_total_kv_duplicates" ] || first_total_kv_duplicates=$(( first_total_kv_faults - first_total_kv_after_dedup ))
  [ -n "$last_total_duplicates" ] || last_total_duplicates=$(( last_total_faults - last_total_after_dedup ))
  [ -n "$last_total_kv_duplicates" ] || last_total_kv_duplicates=$(( last_total_kv_faults - last_total_kv_after_dedup ))

  for v in \
    "$first_batch_faults" "$first_batch_duplicates" "$first_batch_after_dedup" \
    "$first_batch_kv_faults" "$first_batch_kv_duplicates" "$first_batch_kv_after_dedup" \
    "$first_total_faults" "$first_total_duplicates" "$first_total_after_dedup" \
    "$first_total_kv_faults" "$first_total_kv_duplicates" "$first_total_kv_after_dedup" \
    "$last_total_faults" "$last_total_duplicates" "$last_total_after_dedup" \
    "$last_total_kv_faults" "$last_total_kv_duplicates" "$last_total_kv_after_dedup"
  do
    [ -n "$v" ] || {
      echo "===== Delta Replayable fault stats (this workload) ====="
      echo "Failed to parse stats lines from $stats_file"
      echo "Hint: keep machine-readable key=value fields in driver stats logs (e.g., batch_faults=..., total_faults=...)."
      return 0
    }
  done

  # Baseline at benchmark start:
  # baseline = first_total - first_batch
  local base_total_faults base_total_duplicates base_total_after_dedup
  local base_total_kv_faults base_total_kv_duplicates base_total_kv_after_dedup
  base_total_faults=$(( first_total_faults - first_batch_faults ))
  base_total_duplicates=$(( first_total_duplicates - first_batch_duplicates ))
  base_total_after_dedup=$(( first_total_after_dedup - first_batch_after_dedup ))
  base_total_kv_faults=$(( first_total_kv_faults - first_batch_kv_faults ))
  base_total_kv_duplicates=$(( first_total_kv_duplicates - first_batch_kv_duplicates ))
  base_total_kv_after_dedup=$(( first_total_kv_after_dedup - first_batch_kv_after_dedup ))

  local delta_faults delta_duplicates delta_after_dedup
  local delta_kv_faults delta_kv_duplicates delta_kv_after_dedup
  delta_faults="$(safe_non_negative_delta "$last_total_faults" "$base_total_faults")"
  delta_duplicates="$(safe_non_negative_delta "$last_total_duplicates" "$base_total_duplicates")"
  delta_after_dedup="$(safe_non_negative_delta "$last_total_after_dedup" "$base_total_after_dedup")"
  delta_kv_faults="$(safe_non_negative_delta "$last_total_kv_faults" "$base_total_kv_faults")"
  delta_kv_duplicates="$(safe_non_negative_delta "$last_total_kv_duplicates" "$base_total_kv_duplicates")"
  delta_kv_after_dedup="$(safe_non_negative_delta "$last_total_kv_after_dedup" "$base_total_kv_after_dedup")"

  local delta_kv_ratio delta_kv_after_dedup_ratio
  delta_kv_ratio="$(calc_ratio_percent "$delta_kv_faults" "$delta_faults")"
  delta_kv_after_dedup_ratio="$(calc_ratio_percent "$delta_kv_after_dedup" "$delta_after_dedup")"

  echo "===== Delta Replayable fault stats (this workload) ====="
  echo "formula: baseline = first_total - first_batch; delta = last_total - baseline"
  echo "delta_faults=$delta_faults delta_duplicates=$delta_duplicates delta_after_dedup=$delta_after_dedup"
  echo "delta_kv_faults=$delta_kv_faults delta_kv_duplicates=$delta_kv_duplicates delta_kv_after_dedup=$delta_kv_after_dedup"
  echo "delta_kv_ratio=${delta_kv_ratio}% delta_kv_after_dedup_ratio=${delta_kv_after_dedup_ratio}%"
}

main() {
  parse_args "$@"

  trap cleanup EXIT INT TERM

  if [ "$NO_CLEANUP" -eq 0 ]; then
    python3 "$REPO_ROOT/workloads/cleanup_gpu.py" || true
    sleep 2
  fi

  if [ "$RESET_COUNTERS" -eq 1 ]; then
    reload_uvm_module_for_counter_reset
  fi

  validate_inputs

  echo "Starting single-process vLLM server..."
  start_server

  echo "Waiting for server readiness + fresh KV range..."
  wait_for_server_and_kv_range

  echo "KV range (same server process): start=$KV_START end=$KV_END"
  configure_uvm_params

  echo "UVM params configured."
  echo "Stats log: $TRACE_LOG"
  [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] && echo "Address log: $ADDRESS_TRACE_LOG"

  start_trace_capture_if_needed
  start_dmesg_capture_if_needed

  if [ "$NO_BENCH" -eq 0 ]; then
    echo "Running benchmark against same server process:"
    printf '  %q' "${BENCH_CMD[@]}"
    echo
    echo "Bench log: $BENCH_LOG"
    echo "Benchmark started. Waiting for completion..."

    set +e
    "${BENCH_CMD[@]}" 2>&1 | tee "$BENCH_LOG"
    bench_status=$?
    set -e

    echo "Benchmark exit code: $bench_status"
    if [ "$bench_status" -ne 0 ]; then
      echo "===== bench log tail ====="
      tail -n 80 "$BENCH_LOG" 2>/dev/null || true
      die "Benchmark command failed"
    fi
  else
    echo "--no-bench enabled: server started and UVM params configured, benchmark skipped."
  fi

  stop_dmesg_capture_if_needed
  stop_trace_capture_if_needed

  print_recent_stats
  print_delta_stats
  if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
    echo "Allocator trace log: $ALLOCATOR_TRACE_LOG"
  fi
}

main "$@"
