# vLLM 固定赋分策略改造完整文档（2026-04-16）

## 1. 文档目的

本文档完整记录本次项目改造：

1. 停止使用 KVCache 重要性推断（如 StreamingLLM 启发式）进行预取/驱逐辅助决策。
2. 改为由 vLLM 基于地址位置与内存类别（KV / Weights）进行固定分数与 tier 赋值。
3. 将赋值结果写入现有 `attention_score_map`，供 eBPF 侧沿用当前驱逐策略执行。

本文档覆盖：目标、范围、实现方案、逐文件改动、运行方式、验证、兼容性、风险与后续建议。

---

## 2. 背景与变更动机

旧方案的问题：

1. KV 重要性推断依赖启发式（sink/recent），行为受序列形态、并发状态影响较大。
2. 推断逻辑复杂，调参与运维成本高，难以保证不同 workload 下的一致性。
3. 项目当前阶段更需要稳定、可解释、可控的策略边界。

新方案原则：

1. 不再推断 KV 内部“相对重要性”。
2. 只区分内存类别：`KV` 与 `Weights`。
3. 每类采用固定 `score/tier`，由 vLLM 直接赋值。
4. eBPF 侧继续消费 `attention_score_map`，无需先引入新的 map schema。

---

## 3. 变更范围

本次已改动文件：

1. `workloads/vllm/score_bridge_vllm.py`
2. `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py`
3. `workloads/vllm/SCORE_BRIDGE_INTEGRATION.md`
4. `auto_bridge.sh`
5. `docs/vllm_score_bridge_address_driven_changes_20260416.md`（本文档）

本次未改动（保持现状）：

1. `extension/attention_aware_eviction.bpf.c` 的 map key/value 结构。
2. `attention_score_map` 的 key 仍为 `u32 page_id`。
3. `score_bridge.py`（旧工具）未删除，用于监控/兼容。

---

## 4. 目标架构（改造后）

### 4.1 数据流

1. vLLM Worker 初始化 score bridge。
2. `load_model()` 后标注 Weights 对应页面：固定高分高 tier。
3. `initialize_from_config()` 后标注 KV 地址跨度：固定分数固定 tier。
4. 推理过程中周期刷新 KV 页面标注（防止布局变化后失效）。
5. eBPF 在驱逐路径查 `attention_score_map` 并执行既有链表移动策略。

### 4.2 分类与赋分

默认值：

1. KV: `score=20000`，`tier=1 (TIER_COOL)`
2. Weights: `score=65535`，`tier=2 (TIER_HOT)`

可通过环境变量覆盖（见第 7 节）。

---

## 5. 逐文件详细修改

## 5.1 `workloads/vllm/score_bridge_vllm.py`

### 5.1.1 核心行为变化

1. 删除 KV 重要性推断链路（不再使用 StreamingLLM 的 sink/recent/middle 评分逻辑）。
2. 新增固定赋分常量：
   - `DEFAULT_KV_SCORE = 20000`
   - `DEFAULT_KV_TIER = TIER_COOL`
   - `DEFAULT_WEIGHT_SCORE = 65535`
   - `DEFAULT_WEIGHT_TIER = TIER_HOT`
3. 新增/强化类别赋分方法：
   - `hint_kv_cache_layout(...)`
   - `hint_model_weights(model)`
4. `update_from_kv_cache(...)` 保留旧方法名，但内部改为固定赋分实现。
5. `update_from_vllm_worker(...)` 读取 KV 布局后，直接按 KV 固定策略写 map。

### 5.1.2 地址覆盖逻辑

1. 仍按 `2MB` 页面粒度（`VA_SHIFT=21`）转换 `va -> page_id`。
2. KV 标注按 KV 内存跨度覆盖，去重后更新 map。
3. Weights 标注遍历 `named_parameters` + `named_buffers`，按 storage 去重后标注页面。

### 5.1.3 CLI 变化

`daemon` 子命令改为固定赋分参数：

1. `--kv-score`
2. `--kv-tier`
3. `--weight-score`
4. `--weight-tier`

兼容参数（保留但忽略）：

1. `--num-tokens`
2. `--tokens-per-block`

保留这些参数是为了不让现有脚本因为参数不匹配立即失败。

---

## 5.2 `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py`

### 5.2.1 初始化逻辑

在 `Worker.__init__` 中：

1. 当 `score_bridge_vllm` 可导入且 `/sys/fs/bpf/attention_score_map` 存在时启用 bridge。
2. 读取固定赋分配置（环境变量）：
   - `VLLM_SCORE_BRIDGE_KV_SCORE`
   - `VLLM_SCORE_BRIDGE_KV_TIER`
   - `VLLM_SCORE_BRIDGE_WEIGHT_SCORE`
   - `VLLM_SCORE_BRIDGE_WEIGHT_TIER`
3. 读取更新周期：`VLLM_SCORE_BRIDGE_UPDATE_INTERVAL`。
4. `connect()` 后打印启用日志，日志包含四个关键参数值。

### 5.2.2 生命周期集成点

1. `load_model()` 后：
   - `self.score_bridge.hint_model_weights(self.model_runner.model)`
2. `initialize_from_config()` 后：
   - `self.score_bridge.update_from_vllm_worker(self)`
3. `execute_model()`：
   - 每 `N` 步周期刷新 KV 标注
4. `shutdown()`：
   - 停止后台线程（若有）并关闭 bridge

---

## 5.3 `auto_bridge.sh`

### 5.3.1 行为变化

1. 移除旧的日志抓取依赖（不再等待 `DEBUG_KV_PTR`）。
2. 改为通过环境变量 `KV_PTR` 直接输入 KV 基址。
3. 使用新的 daemon 参数启动固定赋分模式。

### 5.3.2 默认参数

脚本默认值：

1. `NUM_BLOCKS=1024`
2. `BLOCK_SIZE_KB=256`
3. `KV_SCORE=20000`
4. `KV_TIER=1`
5. `WEIGHT_SCORE=65535`
6. `WEIGHT_TIER=2`

---

## 5.4 `workloads/vllm/SCORE_BRIDGE_INTEGRATION.md`

已重写为固定赋分方案，删除旧的 StreamingLLM 推断说明，文档现包含：

1. 新策略概述（Position/Category Based）。
2. Worker 嵌入流程。
3. 新环境变量。
4. 新 daemon 命令示例。
5. 监控与排障说明。

---

## 6. 兼容性策略

为降低迁移冲击，本次做了以下兼容处理：

1. 保留 `VLLMScoreBridge` 类名与主要方法名。
2. 保留 `update_from_kv_cache(...)` 方法入口。
3. daemon 保留 `--num-tokens` / `--tokens-per-block` 参数（忽略），避免旧脚本直接报错。
4. `score_bridge.py watch` 监控方式不变。

---

## 7. 配置说明

## 7.1 Worker 内嵌模式环境变量

```bash
export VLLM_SCORE_BRIDGE_KV_SCORE=20000
export VLLM_SCORE_BRIDGE_KV_TIER=1
export VLLM_SCORE_BRIDGE_WEIGHT_SCORE=65535
export VLLM_SCORE_BRIDGE_WEIGHT_TIER=2
export VLLM_SCORE_BRIDGE_UPDATE_INTERVAL=100
export VLLM_SCORE_BRIDGE_VERBOSE=1
```

## 7.2 独立守护模式

```bash
uv run --directory workloads/vllm python workloads/vllm/score_bridge_vllm.py daemon \
  --kv-cache-ptr 0x7f8a12345000 \
  --num-blocks 1024 \
  --block-size-kb 256 \
  --kv-score 20000 \
  --kv-tier 1 \
  --weight-score 65535 \
  --weight-tier 2 \
  --interval 1.0 \
  --stats
```

## 7.3 脚本模式

```bash
export KV_PTR=0x7f8a12345000
bash auto_bridge.sh
```

---

## 8. 验证记录

本次本地验证已执行：

```bash
python3 -m py_compile \
  workloads/vllm/score_bridge_vllm.py \
  workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py
```

结果：通过。

运行时建议检查日志：

1. `Attention-aware score bridge enabled (kv_score=... kv_tier=... weight_score=... weight_tier=...)`
2. `Score bridge marked ... weight pages`
3. `Score bridge initialized KV pages: ...`

---

## 9. 影响分析

## 9.1 正向影响

1. 策略更稳定、可解释，减少启发式误差。
2. 配置更直接，调优只需调整 KV/Weights 固定参数。
3. 项目路径更贴近“按地址语义分类”的长期方向。

## 9.2 代价与限制

1. 失去 KV 内部细粒度差异化驱逐能力（所有 KV 默认同档）。
2. 当前 map key 仍为 `u32 page_id`，不含 `tgid`，跨进程冲突风险仍在。
3. 仍复用现有 score map 语义，尚未引入独立 region map。

---

## 10. 回滚方案

若需回滚到旧推断策略：

1. 回退 `workloads/vllm/score_bridge_vllm.py` 到推断版实现。
2. 回退 `gpu_worker.py` 中固定赋分环境变量与初始化参数。
3. 回退 `SCORE_BRIDGE_INTEGRATION.md` 和 `auto_bridge.sh` 到旧调用参数。

建议以单独分支进行，不要直接在主分支混合回滚与新功能开发。

---

## 11. 后续建议

1. 引入 `tgid + va_block_start` 的 region map，彻底避免跨进程页面冲突。
2. 落地 `uvm_hint_block + uprobe` 语义通道，解耦“分类信息”和“score map”。
3. 在固定赋分基础上支持可选细分层级（例如多档 KV tier），但保持默认简单。
4. 将 `README_SCORE_BRIDGE.md`、`IMPLEMENTATION_SUMMARY.md` 中历史 StreamingLLM 内容同步更新，避免文档歧义。

---

## 12. 结论

本次改造已完成策略切换：

1. 从“KV 重要性推断驱动”切换到“地址/类别固定赋分驱动”。
2. Worker 生命周期接入完整，脚本与文档同步。
3. 保留关键兼容接口，降低迁移风险。

当前版本可作为后续 region-aware 语义通道改造的稳定过渡基线。

