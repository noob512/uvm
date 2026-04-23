# vLLM UVM Policy Stage 0 / Stage 1 Implementation

## 1. 目标

本文档记录本次对 vLLM UVM allocator 的阶段 0 和阶段 1 评估与实现。

阶段定义如下：

1. 阶段 0：trace-only 建模
   - 不改变分配行为
   - 运行 `classify_allocation(...)`
   - 输出 `TRACE_POLICY`
2. 阶段 1：对 warmup workspace 做保守优化
   - 仍然走 `cudaMallocManaged`
   - 对命中 `warmup_workspace` 的 allocation 立即执行 `cudaMemPrefetchAsync(..., device)`
   - 可选附加 `cudaMemAdviseSetPreferredLocation(device)`

## 2. 阶段 0 是否已经实现

结论：

阶段 0 在本次修改前并没有完整实现。

修改前已经具备：

1. `TRACE_PHASE`
2. `TRACE_ALLOC`
3. `TRACE_FREE`
4. phase 标注：
   - `load_model`
   - `initialize_kv_cache`
   - `profile_run`
   - `kernel_warmup:*`
   - `uvm_enable:cublas_preinit`
   - `enabled`

但修改前缺少：

1. `classify_allocation(...)`
2. `AllocationClass`
3. `PolicyAction`
4. `TRACE_POLICY`
5. 显式的 policy mode

所以修改前只能说“具备阶段 0 的原始 trace 基础设施”，不能说阶段 0 已完整实现。

本次修改后，阶段 0 已实现为默认行为：

1. 默认 `VLLM_UVM_POLICY_ENABLE=1`
2. 默认 `VLLM_UVM_POLICY_MODE=trace_only`
3. 每次 allocation 后会输出 `TRACE_POLICY`
4. 不改变 allocation API
5. 不改变 placement 行为

## 3. 阶段 0 实现内容

修改文件：

1. [uvm_allocator.cpp](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp)
2. [run_kv_fault_ratio.sh](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/run_kv_fault_ratio.sh)

### 3.1 新增语义分类

新增 `AllocationClass`：

1. `weight_persistent`
2. `kv_persistent`
3. `warmup_workspace`
4. `runtime_scratch`
5. `runtime_workspace`
6. `unknown_managed`

### 3.2 新增策略动作

新增 `PolicyAction`：

1. `managed_default`
2. `managed_prefetch_gpu`

阶段 0 默认只会输出 `managed_default`，不改变行为。

### 3.3 新增 `classify_allocation(...)`

第一版启发式规则：

1. `phase=load_model`
   - `weight_persistent`
2. `phase=initialize_kv_cache`
   - `kv_persistent`
3. phase 包含以下关键词：
   - `warmup`
   - `autotune`
   - `preinit`
   - 或 `phase=profile_run`
   - 分类为 `warmup_workspace`
4. `phase=enabled` 且 size 在 `1 MiB .. 16 MiB`
   - `runtime_scratch`
5. `phase=enabled` 且 size 在 `16 MiB .. 128 MiB`
   - `runtime_workspace`
6. 其他：
   - `unknown_managed`

### 3.4 新增 `TRACE_POLICY`

每次 allocation 会输出一行：

```text
TRACE_POLICY alloc_id=<id> ptr=<start> end=<end> size_bytes=<n>
size_bucket=<bucket> device=<gpu> phase=<phase>
predicted_class=<class> action=<action>
action_success=<0|1> action_error=<err>
```

该日志用于验证：

1. 阶段 0 是否能稳定识别热点对象
2. 是否能把现有 unknown gap 对应的 allocation 归到：
   - `warmup_workspace`
   - `runtime_scratch`
   - `runtime_workspace`

## 4. 阶段 1 实现内容

阶段 1 已实现，但默认不启用改变行为。

启用条件：

1. `VLLM_UVM_POLICY_ENABLE=1`
2. `VLLM_UVM_POLICY_MODE=prefetch`
   - 或 `warmup_prefetch`
3. allocation 被分类为 `warmup_workspace`
4. allocation size 大于等于 `VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES`

满足上述条件后：

1. 仍然使用 `cudaMallocManaged`
2. 分配后执行 `cudaMemPrefetchAsync(ptr, size, device, stream)`
3. 如果开启 `VLLM_UVM_POLICY_WARMUP_ADVISE_GPU=1`
   - 额外执行 `cudaMemAdviseSetPreferredLocation(device)`

也就是说，阶段 1 不改变 allocator API 契约。

## 5. 为什么阶段 1 默认不启用

默认模式仍是：

```text
trace_only
```

原因：

1. 需要先确认 `TRACE_POLICY` 的分类是否稳定
2. 避免直接改变已有实验 baseline
3. 避免 warmup prefetch 在显存压力较大时引入额外干扰
4. 保留可对照实验：
   - trace-only baseline
   - prefetch enabled

## 6. 新增环境变量

### 6.1 `VLLM_UVM_POLICY_ENABLE`

默认：

```text
1
```

作用：

1. 是否开启 policy 分类日志

### 6.2 `VLLM_UVM_POLICY_MODE`

默认：

```text
trace_only
```

可选：

1. `trace_only`
   - 只输出 `TRACE_POLICY`
   - 不改变行为
2. `prefetch`
   - 对 `warmup_workspace` 执行 prefetch
3. `warmup_prefetch`
   - 等价于更明确的 warmup-only prefetch 模式

### 6.3 `VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES`

默认：

```text
1048576
```

作用：

1. 只有 size 大于等于该阈值的 `warmup_workspace` 才会触发 prefetch

### 6.4 `VLLM_UVM_POLICY_WARMUP_ADVISE_GPU`

默认：

```text
0
```

作用：

1. 是否在 warmup prefetch 前额外执行：
   - `cudaMemAdviseSetPreferredLocation(device)`

建议初期保持为 `0`。

## 7. `run_kv_fault_ratio.sh` 新增参数

新增参数：

1. `--uvm-policy-enable <0|1>`
2. `--uvm-policy-mode <trace_only|prefetch|warmup_prefetch>`
3. `--uvm-policy-warmup-prefetch-min-bytes <n>`
4. `--uvm-policy-warmup-advise-gpu <0|1>`

默认行为：

```text
--uvm-policy-enable 1
--uvm-policy-mode trace_only
--uvm-policy-warmup-prefetch-min-bytes 1048576
--uvm-policy-warmup-advise-gpu 0
```

## 8. 如何运行阶段 0

阶段 0 不改变行为，只收集策略分类日志。

推荐命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_stage0.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-policy-enable 1 \
  --uvm-policy-mode trace_only
```

验证：

```bash
rg 'TRACE_POLICY' /tmp/vllm_uvm_allocator_trace_stage0.log | head
```

期望：

1. 出现 `TRACE_POLICY`
2. `action=managed_default`
3. 不出现 prefetch 行为

## 9. 如何运行阶段 1

阶段 1 对 warmup workspace 做保守 prefetch。

推荐命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm

./run_kv_fault_ratio.sh \
  --mode trace \
  --with-address-log \
  --trace-log /tmp/uvm_kv_fault_stats_stage1.log \
  --address-trace-log /tmp/uvm_kv_fault_addrs_stage1.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage1.log \
  --uvm-trace-min-bytes 1048576 \
  --uvm-policy-enable 1 \
  --uvm-policy-mode prefetch \
  --uvm-policy-warmup-prefetch-min-bytes 1048576 \
  --uvm-policy-warmup-advise-gpu 0
```

验证：

```bash
rg 'TRACE_POLICY.*predicted_class=warmup_workspace.*action=managed_prefetch_gpu' \
  /tmp/vllm_uvm_allocator_trace_stage1.log | head
```

期望：

1. warmup workspace allocation 出现：
   - `action=managed_prefetch_gpu`
2. 其他类别仍保持：
   - `action=managed_default`

## 10. 如何对比阶段 0 与阶段 1

### 10.1 基础分类

阶段 0：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/analyze_uvm_fault_addresses.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --summary-json /tmp/vllm_fault_classification_summary_stage0.json
```

阶段 1：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/analyze_uvm_fault_addresses.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage1.log \
  --summary-json /tmp/vllm_fault_classification_summary_stage1.json
```

### 10.2 unknown deep dive

阶段 0：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage0.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage0.log \
  --top-gaps 10 \
  --gap-csv /tmp/vllm_unknown_gap_heat_stage0.csv \
  --summary-json /tmp/vllm_fault_deep_dive_summary_stage0.json
```

阶段 1：

```bash
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/deep_dive_uvm_faults.py \
  --address-log /tmp/vllm_uvm_address_regions.log \
  --fault-log /tmp/uvm_kv_fault_addrs_stage1.log \
  --allocator-log /tmp/vllm_uvm_allocator_trace_stage1.log \
  --top-gaps 10 \
  --gap-csv /tmp/vllm_unknown_gap_heat_stage1.csv \
  --summary-json /tmp/vllm_fault_deep_dive_summary_stage1.json
```

### 10.3 重点比较指标

重点比较：

1. `unknown` fault 总数
2. hottest unknown gap 的 fault 数
3. `warmup_workspace` 对应 gap 的 fault 是否下降
4. `kv_cache` 和 `weight` 占比是否更稳定
5. benchmark 是否正常结束

## 11. 成功标准

阶段 0 成功标准：

1. allocator trace 中存在 `TRACE_POLICY`
2. hotspot allocation 能被分类为：
   - `warmup_workspace`
   - `runtime_scratch`
   - `runtime_workspace`
   - `weight_persistent`
   - `kv_persistent`
3. `action=managed_default`
4. 实验结果与旧 baseline 不应有明显行为差异

阶段 1 成功标准：

1. warmup workspace 出现：
   - `action=managed_prefetch_gpu`
2. benchmark 正常完成
3. `unknown` 中 warmup/workspace 相关 gap fault 下降
4. 不引入显著的 GPU OOM 或性能回退

## 12. 风险说明

### 12.1 prefetch 可能增加显存压力

`cudaMemPrefetchAsync` 会主动迁移页面到 GPU。

如果 GPU 内存压力已经很高，可能导致：

1. 迁移开销增加
2. 其他 managed pages 被驱逐
3. fault 模式变化

因此阶段 1 默认只作用于 `warmup_workspace`。

### 12.2 `cudaMemAdviseSetPreferredLocation` 暂不默认开启

`PreferredLocation=GPU` 是更强的 hint。

它可能减少 fault，也可能导致更激进的迁移和驻留竞争。

所以本次默认：

```text
VLLM_UVM_POLICY_WARMUP_ADVISE_GPU=0
```

### 12.3 当前分类仍是启发式

本次 `classify_allocation(...)` 仍是第一版启发式规则。

它依赖：

1. phase
2. size

尚未引入在线 history 和 lifetime 预测。

后续阶段可以继续加入：

1. `(phase, size_bucket)` 历史统计
2. median lifetime
3. reuse count

## 13. 本次修改总结

本次修改完成：

1. 阶段 0：
   - 已实现 `classify_allocation(...)`
   - 已实现 `TRACE_POLICY`
   - 默认 trace-only，不改变行为
2. 阶段 1：
   - 已实现 warmup workspace prefetch
   - 通过 `--uvm-policy-mode prefetch` 显式开启
   - 默认关闭行为改变
3. 运行脚本：
   - 已增加策略相关参数
4. 运行时 allocator：
   - 已重新编译并覆盖到 vLLM 使用的 `uvm_allocator.abi3.so`

## 14. 后续建议

下一步建议先跑两组实验：

1. `trace_only`
2. `prefetch`

保持其他参数完全一致，然后比较：

1. `unknown` 总量
2. hottest gap 热度
3. `warmup_workspace` gap fault 是否下降
4. `runtime_scratch` 是否仍然是主要来源

如果阶段 1 对 warmup gap 有明显收益，再进入阶段 2：

1. 引入在线 lifetime/history
2. 对 `runtime_scratch` 做 GPU scratch pool
