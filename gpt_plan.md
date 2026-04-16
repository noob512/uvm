# vLLM KVCache/Weights 地址驱动分流策略（Prefetch + Eviction）详细设计文档

版本: v0.1（Draft）  
日期: 2026-04-14  
适用仓库: `gpu_ext`  
范围: NVIDIA UVM（`nvidia-uvm`）eBPF struct_ops 内存策略 + vLLM(UVM) workload 语义对接  
约束: 本机无 GPU、项目未编译；本文仅输出设计，不在当前主机尝试实现或运行。

---

## 1. 需求与目标（来自 `my_plan.md`）

`my_plan.md` 的核心意图是两点:

1. 当前主机缺 GPU 且项目未编译，生成方案时不做任何运行尝试（本文遵守）。
2. 策略思想: 延续“按数据类型采用不同驱逐/预取策略”，但放弃“仅通过 block 热度判断是否为权重”的识别方式；改为“直接从 vLLM 获取 KVCache 与模型权重的地址”，通过地址完成 KV/Weight 区分，进而分而治之。

落到可执行的目标定义（Spec）:

- G1: **可靠分类**。在 UVM 模式下，对 page fault 触发的 2MB VA block，能以很低开销判断其属于 `KVCache` 还是 `Weights`（可选再加 `Other`）。
- G2: **分流策略**。基于分类结果，对 `KVCache` 与 `Weights` 采用不同 prefetch / eviction 策略（至少做到 KV 与 Weights “互不误伤”，减少互相驱逐）。
- G3: **不依赖热度做“类型识别”**。热度/频率仍可用于同类内部的保护（例如 weight 内部 T1 保护），但不再用于 KV vs Weights 的分类本身。
- G4: **与现有框架贴合**。复用仓库现有的:
  - `uvm_perf_prefetch_get_hint_va_block` kprobe “补上下文”的模式（见 [prefetch_vllm_phase.bpf.c](D:/github/gpu_ext/extension/prefetch_vllm_phase.bpf.c)、[prefetch_pid_tree.bpf.c](D:/github/gpu_ext/extension/prefetch_pid_tree.bpf.c)）。
  - `bpf_gpu_migrate_range()` + `bpf_wq` 做跨 VA block 预取（见 [prefetch_vllm_phase.bpf.c](D:/github/gpu_ext/extension/prefetch_vllm_phase.bpf.c)、[cross_block_prefetch_plan.md](D:/github/gpu_ext/docs/cross_block_prefetch_plan.md)）。
  - `gpu_block_activate` 驱逐侧 hook（注意 `gpu_block_access` 不可用的限制，见 [extension/README.md](D:/github/gpu_ext/extension/README.md)）。

---

## 2. 背景：为什么“地址分流”优于“热度识别”

已有/曾有方案中，用 block 热度或 fault pattern 去推断“是不是权重”，主要问题:

- P1: **误判不可控**。不同阶段（prefill/decode）、不同并发、不同 batch 形态会改变访问强度，热度阈值很难稳定区分 KV 与 Weights。
- P2: **收敛慢**。热度依赖时间窗口积累，类型识别需要“冷启动”学习，会在关键窗口内（启动、prefill、流量突变）失效。
- P3: **热路径负担**。如果每次 fault 都要维护热度统计、做复杂逻辑，会触碰 struct_ops hot-path 的开销上限。
- P4: **vLLM 天然有语义**。KV cache 与 weights 在 vLLM 内部是明确对象（张量、分配点、生命周期），直接拿地址是“真标签”，比推断更稳。

因此本文选择: **应用侧提供地址真值，内核侧只做 O(1) 分类**。

---

## 3. 总体方案概览

### 3.1 架构图（控制面/数据面）

- 应用侧（vLLM）:
  - 在“权重分配完成后”和“KV cache 分配完成后”，产生地址范围（或 2MB block 集合）的语义 hint。
- 语义通道（推荐走 uprobe 语义函数）:
  - vLLM 调用 `uvm_hint_*()` C ABI 函数（放在 `uvm_allocator.abi3.so` 里，便于 attach uprobe）。
  - eBPF uprobe 程序捕获参数，写入 BPF map（region 元数据）。
- 数据面（UVM struct_ops）:
  - 在 prefetch/evict hook 中按地址查询 region map，执行分流策略。

对应仓库已有的“相同套路”参考:

- Phase 语义通道: `uvm_set_phase(int)`（见 [prefetch_vllm_phase.bpf.c](D:/github/gpu_ext/extension/prefetch_vllm_phase.bpf.c) 与 [docs/cross_block_prefetch_plan.md](D:/github/gpu_ext/docs/cross_block_prefetch_plan.md) 里关于 `uvm_set_phase()` 的描述）。
- 本文扩展: `uvm_hint_kv_range()` / `uvm_hint_weight_range()`（或更通用的 `uvm_hint_range(kind, addr, len)`）。

---

## 4. 关键设计选择（推荐路径）

本方案给出两条可落地路径，推荐优先级如下:

### 路径 A（推荐）：vLLM 主动上报“范围”，BPF map 存“2MB block 分类表”
- vLLM 侧:
  - 计算 KV/Weights 覆盖的 2MB block 集合（去重后），按 block 逐个调用 hint（或通过一次 IPC 批量上报给 root loader）。
- BPF 侧:
  - `region_block_map[(tgid, va_block_start)] -> kind`，struct_ops 查询为 O(1)。
- 优点:
  - 数据面最快，逻辑最简单，几乎不需要“范围查找”或循环。
  - “类型识别”完全由 vLLM 真值提供。
- 代价:
  - 初始化阶段需要上报较多条目（例如 30GB weights 约 15k 个 2MB block），但这是一次性成本。

### 路径 B：vLLM 上报少量“区间”，BPF 数据面做区间匹配
- vLLM 侧:
  - 输出少量区间（例如 `{start,end}`），BPF 侧每次 fault 遍历区间数组判断归属。
- 优点:
  - 上报条目少。
- 风险:
  - 如果权重/kv 的 VA 布局碎片化，区间数会上升，热路径遍历变贵。
  - 需要在 BPF 里做 bounded loop（必须严格控制上限）。

本文后续默认按“路径 A”写详细设计；路径 B 作为备选与 fallback。

---

## 5. Region 元数据模型

### 5.1 Region Kind 定义

建议最少三类:

- `REGION_UNKNOWN = 0`：没有识别信息（fallback 策略）
- `REGION_WEIGHTS = 1`：模型权重（生命周期长、循环访问、应优先保留）
- `REGION_KV = 2`：KV cache（动态增长、近期更重要、适合顺向/跨 block 预取）

可选扩展:

- `REGION_OTHER = 3`：临时 buffer、workspace、非关键数据（可更积极驱逐）
- `REGION_INDEX = 4`：Faiss/其他索引（如果将来做多租户混部）

### 5.2 地址规范化到 2MB VA block

gpu_ext/UVM 策略的天然粒度是 2MB VA block（512 * 4KB pages）。规范化函数:

- `VA_BLOCK_SIZE = 2MB = 1 << 21`
- `va_block_start = addr & ~(VA_BLOCK_SIZE - 1)`
- `va_block_end = (addr + len + VA_BLOCK_SIZE - 1) & ~(VA_BLOCK_SIZE - 1)`

vLLM 上报建议以 `va_block_start` 为单位，避免 BPF 侧做对齐与边界处理。

---

## 6. BPF Map 设计（路径 A）

### 6.1 region_block_map（核心）

用途: 2MB VA block 到类型的 O(1) 查询。

- key 建议包含 `tgid`，防止不同进程 UVA 冲突。
- `va_block_start` 用 `u64`，不要截断为 `u32`（注意仓库里有的实验代码用 `va>>21` 再 mask 到 32 位，那是有碰撞风险的；本设计避免）。

示例结构（伪 C）:

```c
struct region_key {
    __u32 tgid;
    __u32 _pad;
    __u64 va_block_start; // aligned 2MB
};

struct region_val {
    __u8  kind;     // weights/kv/...
    __u8  flags;    // optional
    __u16 reserved;
    __u32 epoch;    // lifecycle cleanup (optional)
};
```

Map 选择:

- `BPF_MAP_TYPE_HASH`
- `max_entries` 估算:
  - 以 Qwen3-30B FP8 29GB weights 为例: `29GB / 2MB ≈ 14848`
  - KV 6GB: `6GB / 2MB ≈ 3072`
  - 合计约 18k，加上碎片与 other，建议 `65536` 作为默认上限（与 [attention_aware_eviction.bpf.c](D:/github/gpu_ext/extension/attention_aware_eviction.bpf.c) 的容量量级一致）。

### 6.2 region_epoch_map（可选）

用途: 解决 PID 重用与“旧条目残留”问题，或支持模型 reload。

- key: `tgid`
- value: `epoch`（单调递增）

vLLM 每次初始化上报前先调用 `uvm_hint_epoch(epoch)` 或 `uvm_hint_begin()`，BPF 记录当前 epoch；写入 `region_block_map` 时带 epoch。数据面查到条目但 epoch 不匹配则视为无效。

### 6.3 region_stats（建议）

用途: 观测分流是否命中，以及策略效果。

- `stat_prefetch_kv_xb`
- `stat_prefetch_weight_xb_skip`
- `stat_evict_weight_protect`
- `stat_evict_kv_unprotected`
- `stat_region_miss`
- `stat_region_entries`（当前条目数，用户态维护也可）

---

## 7. vLLM 侧：如何“直接获取地址”并上报

本仓库 vLLM workload 为 UVM fork，KV cache 分配点明确（见 `initialize_kv_cache_tensors()` 返回的 tensor 字典）。参考位置:

- KV cache 分配: [gpu_model_runner.py](D:/github/gpu_ext/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py) 的 `_allocate_kv_cache_tensors()` / `initialize_kv_cache_tensors()`。
- 模型加载: [gpu_worker.py](D:/github/gpu_ext/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py) 的 `load_model()`。

### 7.1 KV cache 地址提取（确定性强）

KV cache 在 `initialize_kv_cache()` 里拿到 `kv_caches: dict[str, torch.Tensor]`，可做:

- 遍历 `kv_caches.values()`
- 去重（按 `tensor.data_ptr()` 去重，避免 shared KV 多次上报）
- 计算 `addr = data_ptr()`
- 计算 `len = numel() * element_size()`

然后将 `[addr, addr+len)` 覆盖到 2MB block 并上报为 `REGION_KV`。

伪代码（Python 侧概念稿）:

```python
seen = set()
for t in kv_caches.values():
    if isinstance(t, list):  # mamba spec may return list of tensors
        tensors = t
    else:
        tensors = [t]
    for x in tensors:
        ptr = x.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        size = x.numel() * x.element_size()
        hint_region_blocks(kind=REGION_KV, addr=ptr, length=size)
```

### 7.2 Weights 地址提取（两种方式）

Weights 的“直接获取地址”有两类实现策略，按工程复杂度排序:

- 方式 W1（简单但可能条目多）:
  - model load 完成后，遍历 `model.named_parameters()` 与必要的 buffer（embedding、lm_head、kv proj 等）。
  - 以 storage 粒度去重（同一 storage 被多个 tensor view 引用时，只上报一次）。
- 方式 W2（推荐，条目更可控）:
  - 利用“分配阶段语义”：在 model load 前后设置 allocator tag（weights），在 KV 分配阶段设置 allocator tag（kv），由 allocator/trace 自动记录该阶段的大块分配。
  - 这与仓库已有的 `uvm_set_phase()` 思路一致，属于“应用-内核语义通道”的同一类机制（见 [docs/future_work_kernel_system.md](D:/github/gpu_ext/docs/future_work_kernel_system.md) §6.2~§6.3）。

本设计建议先落地 W1 以尽快跑通，再迭代到 W2 降低条目/开销。

---

## 8. 语义通道实现（推荐：uprobe + uvm_allocator.abi3.so）

### 8.1 为什么选 `uvm_allocator.abi3.so`

仓库已有 vLLM phase 检测策略就是将 uprobe 挂到 `uvm_allocator.abi3.so` 的导出符号（见 [prefetch_vllm_phase.c](D:/github/gpu_ext/extension/prefetch_vllm_phase.c)）。优点:

- 不依赖 vLLM C++ 内核（`_C.abi3.so`）符号稳定性。
- 该库在 UVM 模式下必然加载，路径可控（workload README 指导复制生成）。

### 8.2 建议的导出 ABI（最小集合）

为避免“实现逻辑”与“语义通道”耦合，建议导出函数体可以是空实现（仅用于 uprobe 捕获参数），例如:

```c
// kind: 1=WEIGHTS, 2=KV
void uvm_hint_range(int kind, const void* addr, size_t len);

// 初始化/清理（可选）
void uvm_hint_begin(uint32_t epoch);
void uvm_hint_end(uint32_t epoch);
```

vLLM Python 用 ctypes 调用这些函数，BPF uprobe 程序在内核侧更新 map。

路径 A 的“逐 2MB block 上报”可以改成更明确的 ABI:

```c
void uvm_hint_block(int kind, uint64_t va_block_start);
```

这会把复杂度放到 vLLM（计算 block 对齐与去重），让 BPF uprobe 逻辑极简，数据面查询也极简。

### 8.3 BPF uprobe 侧行为（概念）

- 输入: `kind`, `addr/va_block_start`, `len`
- 输出: 更新 `region_block_map`

注意点:

- uprobe 在用户进程上下文触发，此时 `bpf_get_current_pid_tgid()` 返回真实 tgid，适合作为 key。
- 若上报的是 range，需要在用户态展开为 block（推荐不要在 BPF 中展开大 range）。

---

## 9. 数据面：分而治之的策略设计

下面给出一个“保守但有效”的第一版策略组合（不追求一步到位最优，追求稳健与可解释）。

### 9.1 Prefetch 分流（核心目标：KV cross-block，Weights 仅 intra-block）

参考 [docs/cross_block_prefetch_plan.md](D:/github/gpu_ext/docs/cross_block_prefetch_plan.md) §3.2 的设想，替换其中“KV region 检测方法(A/B/C)”为“vLLM 直供地址分类”。

策略:

- 所有 region: `intra-block = always_max`（仓库里验证最稳的基线）
- `REGION_KV`:
  - 允许 cross-VA-block prefetch（direction-aware 或 adjacent-stride）
  - 默认 prefetch ahead = 1 个 VA block（2MB），必要时可按 oversub 或 fault-rate 自适应调整
- `REGION_WEIGHTS`:
  - 禁用 cross-block（避免污染与 displacement）
  - 允许保留 always_max（权重常呈 strided within block，intra-block 已覆盖空间局部性）
- `REGION_UNKNOWN/OTHER`:
  - 默认 always_max 或更保守模式（例如 serving 场景可考虑 `prefetch_serving_adaptive` 的门控思想）

实现落点（BPF 层面的逻辑框架）:

1. kprobe `uvm_perf_prefetch_get_hint_va_block` 把 `va_start/va_end/va_space/owner_tgid` 缓存在 per-cpu map（与 [prefetch_vllm_phase.bpf.c](D:/github/gpu_ext/extension/prefetch_vllm_phase.bpf.c) 相同模式）。
2. `gpu_page_prefetch`:
   - 设置 `result_region = max_prefetch_region`
   - 读取 `va_start`，查 `region_block_map`
   - 若 `kind==KV` 则走 cross-block 调度（`bpf_wq` + `bpf_gpu_migrate_range`），否则跳过

### 9.2 Eviction 分流（核心目标：Weights 优先保留，KV 不与之争抢）

约束来自仓库实践（见 [extension/README.md](D:/github/gpu_ext/extension/README.md)）:

- `gpu_block_access` 基本不可用，驱逐策略主要在 `gpu_block_activate` 做。
- `move_head` 有已知风险（可能触发 Xid 31）；第一版设计尽量只用 `move_tail` 做保护，不做激进“处决”。

第一版 eviction 策略:

- `REGION_WEIGHTS`:
  - 强保护: `move_tail`（等价“更不容易被 evict”）
  - 可选: 延续 cycle_moe 的 T1 阈值逻辑（例如 fault 激活次数 >= 3 才 move_tail），但对权重可更激进（直接 move_tail）以尽快稳定。
- `REGION_KV`:
  - 弱保护或不保护: 不 move_tail（让其自然靠近 head，被优先 evict）
  - 解释: KV 的“优先级”更依赖近期性；如果未来要做更精细，可引入 request/token age，但第一版先做到“别把 weights 挤出去”。
- `REGION_UNKNOWN/OTHER`:
  - 默认行为（return 0），交给 UVM 原始链表逻辑；或按需要弱保护。

这样至少实现了“分而治之”的最小闭环:

- Weights 尽量常驻，减少反复 fault-back。
- KV 作为弹性工作集，GPU 缓存化但不抢占 weights 的生存空间。
- Prefetch 只对 KV 走跨 block，降低 stall，但不会把 weights 赶走。

---

## 10. 回退与失败模式设计

必须考虑 region hint 可能缺失或不完整的情况:

- F1: vLLM 未上报（或 uprobe 未 attach 成功）:
  - `region_map miss` 时一律走保守基线: `always_max + cycle_moe`（仓库默认强基线）。
- F2: 只上报了 KV，Weights 未上报:
  - 将 `KV` 识别为 KV，其余统一视为 Weights 或 Unknown（建议默认 Unknown，仍按 weights 的“intra-only + 强保护”会更稳，但可能误保护 other）。
- F3: map 容量不足（超过 max_entries）:
  - 用户态统计溢出并报警，策略退化为区间匹配或粗粒度范围。
- F4: PID 重用导致旧条目污染:
  - 启用 epoch 机制或在 loader 侧做进程生命周期清理。

---

## 11. 观测与验证（不依赖实现，但要求设计可测）

建议最小可观测闭环:

- region 命中率:
  - `region_hit_kv`, `region_hit_weights`, `region_miss`
- prefetch 分流统计:
  - `xb_sched_kv`, `xb_skip_weight`, `migrate_ok`, `migrate_fail`
- eviction 分流统计:
  - `protect_weight_tail`, `kv_no_protect`, `unknown_default`
- 性能指标（workload 侧）:
  - TTFT、TPOT、throughput
  - page fault 数量与分布（可用现有 trace 工具或 UVM stats）

仓库已有的辅助文档与工具可作为验证参考:

- vLLM region-aware 设想: [cross_block_prefetch_plan.md](D:/github/gpu_ext/docs/cross_block_prefetch_plan.md) §3.2
- KV 生命周期区分设想: [future_work_kernel_system.md](D:/github/gpu_ext/docs/future_work_kernel_system.md) §6
- 现成 phase uprobe 机制: [prefetch_vllm_phase.bpf.c](D:/github/gpu_ext/extension/prefetch_vllm_phase.bpf.c)

---

## 12. 工程拆分与里程碑（仅规划，不在本机实施）

M1: 语义通道打通（地址真值能进入内核）
- 在 `uvm_allocator.abi3.so` 增加 `uvm_hint_*` 导出符号（或采用同等可 uprobe 的稳定符号）。
- vLLM 在 KV cache 分配后上报 KV 地址范围（或 2MB block 列表）。
- vLLM 在 weights load 完成后上报 weights 地址范围（或 2MB block 列表）。
- BPF uprobe 将信息写入 `region_block_map`，并提供 stats 证明命中。

M2: region-aware prefetch（先只改 prefetch）
- 基线 intra-block always_max 保持不变。
- 仅当 `REGION_KV` 时启用 cross-block prefetch。

M3: region-aware eviction（再改 eviction）
- `REGION_WEIGHTS` move_tail 强保护。
- `REGION_KV` 不保护（默认链表行为）。

M4: Serving 场景调参
- 根据 oversub ratio / fault-rate 调整 KV cross-block 深度与速率。
- 若 decode 阶段 cross-block 有害，可结合 phase gating（已有 `uvm_set_phase` 机制）实现 KV-only 且 phase-aware 的 cross-block。

---

## 13. 附录：最小策略伪代码（路径 A）

### 13.1 vLLM 上报（概念）

```python
VA_BLOCK = 2 * 1024 * 1024

def blocks_covered(addr: int, length: int) -> set[int]:
    start = addr & ~(VA_BLOCK - 1)
    end = (addr + length + VA_BLOCK - 1) & ~(VA_BLOCK - 1)
    return {x for x in range(start, end, VA_BLOCK)}

def hint_tensor(kind: int, tensor: torch.Tensor):
    addr = tensor.data_ptr()
    length = tensor.numel() * tensor.element_size()
    for b in blocks_covered(addr, length):
        uvm_hint_block(kind, b)  # C ABI, uprobe 捕获
```

### 13.2 BPF 数据面分流（概念）

```c
// kprobe: uvm_perf_prefetch_get_hint_va_block
cache.va_start = va_block->start;
cache.va_end = va_block->end;
cache.va_space = va_block->managed_range->va_range.va_space;
cache.owner_tgid = get_owner_pid_from_va_block(va_block);

// struct_ops: gpu_page_prefetch
always_max(result_region);

kind = lookup(region_block_map, (owner_tgid, cache.va_start));
if (kind == REGION_KV) {
    if (direction_forward(history)) {
        schedule_bpf_wq_migrate(cache.va_space, cache.va_end /* next block */, 2MB);
    }
}
```

---

如果你希望我把这份文档进一步“落到仓库约定的文件形态”（例如按你们 docs 目录的结构写成 `docs/gpu-ext/policy/` 下的新 markdown，并对照已有 `future_work_kernel_system.md` / `cross_block_prefetch_plan.md` 做一致的术语与图示风格），你告诉我希望放置的路径与文档命名规则即可；我仍然只写文档，不做任何实现与运行。