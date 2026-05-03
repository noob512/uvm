# vLLM UVM Stage K Global Pool Coordinator 实现说明

Stage K 当前实现的是全局 high-level action coordinator：在已经安全接入的运行时动作点上统一预算和统一 trace。它不在 allocator 层按裸指针迁移对象，也不声称已经完成完整 KV cache CPU swap/prefetch。

## 1. 当前完成内容

1. 新增 `vllm.device_allocator.uvm_pool_coordinator`。
2. 支持 `trace_only|enforce` 两种模式。
3. 支持全局 action budget 和 per-pool action budget。
4. Stage I expert weight GPU prefetch / cold expert advise / cold expert CPU prefetch 发起前会请求 coordinator grant。
5. Stage J KV runtime pressure 会上报 coordinator pressure。
6. Stage J prefix-cache free-block eviction executor 发起前会请求 coordinator grant。
7. 输出统一 Stage K JSONL trace。

## 2. 环境变量

```text
VLLM_UVM_POOL_COORDINATOR_ENABLE=0|1
VLLM_UVM_POOL_COORDINATOR_MODE=trace_only|enforce
VLLM_UVM_POOL_COORDINATOR_TRACE_FILE=/path/to/vllm_uvm_pool_coordinator_stage_k.jsonl
VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP=0
VLLM_UVM_POOL_COORDINATOR_PRIORITY=kv,weights,scratch
```

`0` 表示不限额。`trace_only` 下即使超预算也允许动作继续执行，但会记录 `would_deny=true`。`enforce` 下超预算动作会被跳过。

## 3. Runner 参数

`run_kv_fault_ratio.sh` 增加：

```bash
--uvm-pool-coordinator-enable 1
--uvm-pool-coordinator-mode trace_only
--uvm-pool-coordinator-trace-file /tmp/vllm_uvm_pool_coordinator_stage_k.jsonl
--uvm-pool-coordinator-global-bytes-per-step 1048576
--uvm-pool-coordinator-weight-bytes-per-step 1048576
--uvm-pool-coordinator-kv-bytes-per-step 1048576
--uvm-pool-coordinator-scratch-bytes-per-step 1048576
--uvm-pool-coordinator-priority kv,weights,scratch
```

## 4. Trace 格式

主要 action：

```text
coordinator_config
coordinator_pressure
coordinator_request
coordinator_summary
coordinator_scope_reset
```

`coordinator_request` 关键字段：

```text
pool
requested_action
requested_bytes
allowed
would_deny
reason
scope_key
pool_budget_bytes
pool_used_bytes
global_budget_bytes
global_used_bytes
metadata
```

## 5. 与 Stage I 的关系

Stage I 仍负责在 MoE layer 安全点执行 expert-only actions：

1. active hot expert GPU prefetch。
2. cold inactive expert CPU preferred-location advise。
3. cold inactive expert CPU prefetch。

Stage K 只决定这些 action 是否获得本 step 的全局/weights 预算。Stage K 不改变 Stage H plan，不改变 expert 选择逻辑。

## 6. 与 Stage J 的关系

Stage J 仍负责 KV block-manager 边界的 runtime pressure 和 safe prefix-cache free-block eviction。

Stage K 当前只协调 prefix-cache eviction 这个安全 executor 的预算。它不负责：

1. 活跃 KV block CPU swap。
2. KV block reload/prefetch-back。
3. 修改 active request block table。
4. 直接迁移 KV tensor raw pointer。

这些仍属于后续 Stage J2/J3 或 KV connector/offload 接入。

## 7. 验收脚本

快速 self-test：

```bash
cd workloads/vllm
./check_stage_k_success.py
```

这个测试不启动 GPU server，会构造 weights、KV、scratch 三类 action 请求，并用小预算强制出现 `would_deny`。

端到端 GPU probe：

```bash
cd workloads/vllm
./check_stage_k_success.py --gpu-run --mode trace_only
```

GPU probe 会先生成 Stage H weight expert plan，再同时开启 Stage I、Stage J、Stage K，要求：

1. Stage K trace 存在。
2. coordinator config 存在且 enabled。
3. 至少 weights 和 KV pool 出现在统一 trace 中。
4. 有 coordinator request。
5. 有 would-deny 记录。
6. trace-only 模式下 benchmark failed requests 为 0。

## 8. 当前边界

Stage K 第一版的价值是把多 pool 策略从“各自为战”推进到“统一预算与统一报告”。它还不是最终的跨 pool 数据迁移管理器。下一步如果继续增强，应优先补：

1. Stage J2/J3：真实 KV offload/load/prefetch-back。
2. allocator/C++ 侧 scratch pool 与 Stage K 的直接 budget hook。
3. 按时间窗口而非单 scope 的 bandwidth budget。
4. 更细的 priority arbitration。
