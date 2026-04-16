# vLLM UVM 实验异常分析与 eBPF 卸载自检说明

## 1. 背景与约束

本文回答 [`问题.txt`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/问题.txt) 中提出的 7 个问题，并记录本次代码改动。

明确约束：

- 按要求，**没有重新运行**任何耗时 benchmark。
- 结论基于已有结果文件、vLLM benchmark 脚本、extension loader、内核驱动代码，以及现有文档。
- 文中会区分两类内容：
  - **代码事实**：可以直接从源码确认。
  - **原因推断**：根据现象和代码推导出的最合理解释。

---

## 2. 结论先行

### 2.1 简短结论

`serve_bench.py` **不是造成这组异常现象的主要原因**。它确实只是启动 vLLM、调用 `vllm bench serve`、解析 summary，并把原始输出完整写入 JSON。你看到的 40% 左右性能差异，不像是“统计脚本写错了”，而更像是**实验状态泄漏**。

当前证据下，**最强解释**是：

1. **eBPF struct_ops 没有被完全卸载**，导致你以为是在跑“纯 UVM baseline”，实际上仍然在走已挂载的 BPF policy。
2. loader 侧过去**没有在启动前/退出后验证 struct_ops 是否真的清空**，因此实验状态不可观测。
3. NVIDIA UVM 驱动中的 BPF policy 是通过一个**全局单例 `uvm_ops` 指针**生效的；如果旧 policy 残留，新的实验结论就会被污染。

### 2.2 对现象的解释

对你给出的 4 组结果，最合理的解释是：

- “重启后第一次跑纯 UVM 很差”：
  - 这很像**真正的纯 UVM baseline**。
- “之后再挂 eBPF，结果仍然没改善”：
  - 这说明“你以为已经挂上的 policy”当时**很可能没有真正接管到驱动路径**，或者实验状态没有被正确切换。
- “如果第一次就先挂 eBPF，性能显著改善”：
  - 这符合 BPF prefetch/eviction policy 生效后的预期。
- “卸载 eBPF 后再跑纯 UVM，性能仍然保持改善”：
  - 这**最像卸载不完整**，即所谓“纯 UVM”其实仍然带着残留的 struct_ops。

最后这一条，是当前证据里最关键、也最有辨识度的异常。

---

## 3. 先检查 `serve_bench.py`：记录程序本身有没有问题

检查文件：[`serve_bench.py`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/configs/serve_bench.py)

### 3.1 这段脚本实际做了什么

脚本流程非常直接：

1. 根据 `--mode` 组织 vLLM server 启动命令。
2. 对 `uvm` 模式仅设置环境变量 `VLLM_USE_UVM=1`。
3. 等 server ready 后，调用 `uv run vllm bench serve ...`。
4. 用 [`common.py`](/home/ubuntu/nvidia-uvm-gpu/workloads/scripts/common.py) 里的 `parse_vllm_bench_output()` 从 benchmark summary 中提取指标。
5. 同时把完整 `bench_output` 原样写进结果 JSON 的 `raw.bench_output`。

### 3.2 为什么我判断它不是主因

有三个理由：

1. **它解析的是 vLLM benchmark 自己打印出来的 summary**
   - 不是自造指标。
   - `successful_requests`、`Benchmark duration`、`TTFT`、`TPOT` 都来自官方 bench 输出。

2. **原始输出被完整保存在 JSON 里**
   - 例如 [`uvm_baseline.json`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/results/exp3/uvm_baseline.json) 和 [`uvm_ebpf_base_3.json`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/results/exp3/uvm_ebpf_base_3.json) 中，`raw.bench_output` 与 `metrics` 是对得上的。
   - 所以这里不是“脚本把 145806 ms 解析成 254915 ms”这类错误。

3. **异常模式不是统计 bug 的典型表现**
   - 如果是记录脚本 bug，通常会表现为字段错位、数值单位错误、某些 run 全为 0、或随机不一致。
   - 现在的现象是“是否先挂 eBPF 决定了后续多次实验的性能状态”，这更像**系统状态残留**，不像 JSON 记录错误。

### 3.3 但它确实有一个“可观测性缺口”

虽然 `serve_bench.py` 没有伪造数据，但它**没有记录当前系统到底是不是纯 UVM**。它不会回答下面这些关键问题：

- 当前是否还有 UVM struct_ops 残留？
- 当前挂的是哪个 policy？
- attention-aware 的 pinned maps 是否还存在？
- “这次 baseline”是不是其实还带着旧 BPF 状态？

所以：

- **结论**：`serve_bench.py` 不是主 bug。
- **但**：它以前缺少“实验状态元数据”，这会放大误判风险。

---

## 4. 结果文件本身说明了什么

检查文件：

- [`uvm_baseline.json`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/results/exp3/uvm_baseline.json)
- [`uvm_ebpf_base.json`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/results/exp3/uvm_ebpf_base.json)
- [`uvm_ebpf_base_3.json`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/results/exp3/uvm_ebpf_base_3.json)
- [`uvm_base_2.json`](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/results/exp3/uvm_base_2.json)

### 4.1 关键对比

| 场景 | Benchmark duration (s) | Mean TTFT (ms) | Mean TPOT (ms) | Output tok/s |
|---|---:|---:|---:|---:|
| 重启后第一次纯 UVM | 674.87 | 254915.61 | 191.30 | 74.88 |
| 先跑差的 UVM，再挂 eBPF | 660.75 | 248978.81 | 186.16 | 76.48 |
| 重启后第一次就先挂 eBPF | 392.72 | 145806.31 | 109.48 | 129.47 |
| 先跑好 eBPF，再“卸载”后跑纯 UVM | 418.93 | 151829.55 | 117.82 | 121.37 |

### 4.2 差异幅度

以第一次差的纯 UVM 作为基线：

- “差 UVM -> 差状态下再挂 eBPF”
  - `benchmark_duration_s`: 仅改善 **2.09%**
  - `mean_ttft_ms`: 仅改善 **2.33%**
  - `mean_tpot_ms`: 仅改善 **2.69%**

- “差 UVM -> 第一次就挂 eBPF”
  - `benchmark_duration_s`: 改善 **41.81%**
  - `mean_ttft_ms`: 改善 **42.80%**
  - `mean_tpot_ms`: 改善 **42.77%**
  - `output_throughput_tok_s`: 提升 **72.90%**

- “差 UVM -> 好 eBPF 后再跑所谓 baseline”
  - `benchmark_duration_s`: 仍改善 **37.92%**
  - `mean_ttft_ms`: 仍改善 **40.44%**
  - `mean_tpot_ms`: 仍改善 **38.41%**

这组数据最难解释的一点是：

> 如果 eBPF 真被完全卸载，那么最后一组“纯 UVM”不应该仍然和 eBPF 组几乎一样好。

---

## 5. 结合驱动和文档，详细解释上述结果

这一节分成“代码事实”和“最合理推断”。

### 5.1 代码事实 1：UVM struct_ops 在驱动里是全局单例

检查文件：[`uvm_bpf_struct_ops.c`](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c)

关键实现：

- 驱动中有一个全局指针：
  - `static struct gpu_mem_ops __rcu *uvm_ops;`
- 注册时：
  - `cmpxchg(&uvm_ops, NULL, ops)`
- 注销时：
  - `cmpxchg(&uvm_ops, ops, NULL)`

这意味着：

1. **同一时刻只能有一个 UVM policy 生效**。
2. 所有 prefetch / eviction hook 最终都通过这个全局 `uvm_ops` 指针被调用。
3. 如果 `uvm_ops` 没回到 `NULL`，那就**不是纯 UVM**。

这和文档里的说明一致：

- [`cross_block_prefetch_plan.md`](/home/ubuntu/nvidia-uvm-gpu/docs/cross_block_prefetch_plan.md) 明确写了“同一时间只能加载一个 BPF struct_ops 策略”。

### 5.2 代码事实 2：当前 loader 过去并不会确认“旧 policy 真的没了”

检查文件：

- [`prefetch_adaptive_sequential.c`](/home/ubuntu/nvidia-uvm-gpu/extension/prefetch_adaptive_sequential.c)
- [`attention_aware_eviction.c`](/home/ubuntu/nvidia-uvm-gpu/extension/attention_aware_eviction.c)
- [`cleanup_struct_ops.h`](/home/ubuntu/nvidia-uvm-gpu/extension/cleanup_struct_ops.h)

原先的问题有两个：

1. 两个 loader 都会调用 `cleanup_old_struct_ops()`，但**不检查返回值**。
2. 退出时它们会 `bpf_link__destroy(link)`，但**不会再扫一遍系统确认 UVM struct_ops 是否真的为 0**。

这会带来两个后果：

- 启动前如果已经有残留实例，程序可能继续执行，实验状态不干净。
- 退出后即使残留没清空，用户也看不到明确报错。

### 5.3 代码事实 3：文档里已经多次把“强制清理 struct_ops 残留”当成必要步骤

检查文件：

- [`cross_block_prefetch_plan.md`](/home/ubuntu/nvidia-uvm-gpu/docs/cross_block_prefetch_plan.md)
- [`phase1_score_aware_eviction.md`](/home/ubuntu/nvidia-uvm-gpu/docs/phase1_score_aware_eviction.md)

文档中已经反复出现下面的经验：

- 切换策略前要清理旧 loader。
- 需要跑 `cleanup_struct_ops_tool`。
- 甚至准备了 “force-clean struct_ops” 的额外兜底脚本。

这说明在项目已有经验里，**struct_ops 残留不是理论风险，而是已经被当成真实实验风险在处理**。

### 5.4 代码事实 4：UVM 的 eviction / LRU 状态确实是驱动级全局状态

检查文件：

- [`UVM_LRU_USAGE_GUIDE.md`](/home/ubuntu/nvidia-uvm-gpu/docs/gpu-ext/driver_docs/lru/UVM_LRU_USAGE_GUIDE.md)
- [`uvm_pmm_gpu.c`](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_pmm_gpu.c)

重要事实：

- GPU PMM 里维护全局 `root_chunks.va_block_used` / `va_block_unused` 链表。
- chunk 在 unpin / allocate / evict 时会更新这些全局链表。
- `gpu_block_activate` hook 就插在 chunk 进入 evictable 状态时。

所以即使不考虑“卸载失败”，UVM 的 PMM/LRU 行为本来就不是“每次跑脚本都完全从零开始”的用户态对象，而是**模块级、GPU 级状态机**。

### 5.5 最强推断：你看到的主要问题是“卸载不完整导致假 baseline”

这是当前最合理的主结论。

证据链是闭合的：

1. 纯 UVM baseline 很差。
2. 如果第一次就挂 eBPF，性能显著变好。
3. 之后即使“卸载”再跑 baseline，依然大幅好于真正 baseline。
4. 驱动里 `uvm_ops` 是全局单例，只要没归零，就仍在走 BPF path。
5. 原 loader 没有在退出后验证这一点。

因此最自然的解释就是：

> 你后面看到的“无 eBPF 纯 UVM”，至少有一部分 run 并不是真正的纯 UVM，而是带着残留 struct_ops 的“伪 baseline”。

### 5.6 为什么“先跑差 baseline，再挂 eBPF 仍然没改善”？

这一点不能像上一条那样 100% 由代码直接证明，所以这里明确标为**推断**。

我认为有两种可能，且第一种更值得优先排查：

#### 可能性 A：当时“挂载成功”这件事本身没有被证明

也就是说，当时虽然你执行了 loader，但没有一个可靠机制告诉你：

- 新 policy 是否真的 attach 成功；
- attach 成功后系统里是否只剩它一个 UVM struct_ops；
- 旧实例是否完全消失；
- attention-aware 的 pinned maps 是否还残留。

由于旧代码没有做启动前/退出后自检，这种不确定性是客观存在的。

#### 可能性 B：第一次 run 改变了驱动级内存状态，后续 run 受到了历史状态影响

这条也有一定理论基础：

- UVM 的 PMM / root chunk 列表是模块级状态。
- prefetch 与 eviction hook 都会影响 chunk 何时进入 used/unused 链表以及如何被排序。
- 文档也反复说明 UVM 的性能瓶颈与 fault path、chunk thrashing、prefetch 范围选择强相关。

但是要注意：

- 这种“历史状态影响后续 run”更像**次级放大因素**。
- 它不足以像“卸载不完整”那样，直接解释“明明卸了 eBPF，baseline 还和 eBPF 一样快”。

所以我的判断是：

- **主因**：struct_ops 残留导致实验状态没有真正切换。
- **次因**：即使将来把卸载做对，UVM/PMM 的全局状态也仍可能让“第一次运行”与“后续运行”之间存在轻微历史依赖。

---

## 6. 为什么说这不是 `serve_bench.py` 的统计假象

再强调一次，因为这是第 4 条问题里的核心检查项。

### 6.1 `parse_vllm_bench_output()` 的行为是正常的

检查文件：[`common.py`](/home/ubuntu/nvidia-uvm-gpu/workloads/scripts/common.py)

它只是用正则抓这些行：

- `Successful requests`
- `Benchmark duration (s)`
- `Mean TTFT (ms)`
- `Mean TPOT (ms)`
- 以及吞吐量相关字段

这些值与 `raw.bench_output` 中 summary 完全对应。

### 6.2 即使解析稍有瑕疵，也不可能制造这种状态依赖现象

就算我们假设解析逻辑有小问题，它也顶多会影响：

- 少数字段缺失；
- 个别数值格式兼容性；
- stdout/stderr 拼接顺序。

但它不可能制造下面这种逻辑现象：

- “第一次挂 eBPF 就快”
- “先不挂就慢”
- “卸载后还继续快”

这已经超出“日志解析错误”的范畴，属于**系统状态切换出了问题**。

---

## 7. 本次代码改动

### 7.1 改动目标

由于 `bpftool` 当前不可用，本次改动的目标不是“修复所有实验现象”，而是先解决**可观测性缺失**：

- 让 loader 在运行时自己告诉你：
  - 启动前是否已经是干净状态；
  - 退出后是否真的没有残留 struct_ops；
  - attention-aware 的 pinned maps 是否真的删掉了。

### 7.2 改动文件

- [`cleanup_struct_ops.h`](/home/ubuntu/nvidia-uvm-gpu/extension/cleanup_struct_ops.h)
- [`prefetch_adaptive_sequential.c`](/home/ubuntu/nvidia-uvm-gpu/extension/prefetch_adaptive_sequential.c)
- [`attention_aware_eviction.c`](/home/ubuntu/nvidia-uvm-gpu/extension/attention_aware_eviction.c)

### 7.3 具体改了什么

#### A. `cleanup_struct_ops.h`

新增了 3 个共享辅助能力：

1. `collect_uvm_struct_ops_instances(...)`
   - 扫描当前系统里所有 `uvm_ops*` 的 struct_ops map。

2. `print_uvm_struct_ops_instances(...)`
   - 把残留实例的 `id` 和 `name` 明确打印出来。

3. `verify_no_uvm_struct_ops_instances(...)`
   - 用于做“干净状态断言”。
   - 如果还有实例残留，直接返回 `-EEXIST`。

同时保留了原有 `cleanup_old_struct_ops()`，但现在它在发现残留时会把实例列表打印得更清楚。

#### B. `prefetch_adaptive_sequential.c`

新增了两段关键逻辑：

1. **启动前检查**
   - 调用 `cleanup_old_struct_ops()` 后，不再忽略返回值。
   - 如果还有残留实例，直接拒绝启动。
   - 再调用 `verify_no_uvm_struct_ops_instances("startup")` 做二次确认。

2. **退出后检查**
   - `bpf_link__destroy(link)` 之后，立刻再次扫描。
   - 如果系统里仍然存在 UVM struct_ops，程序会明确报错并返回非 0。

这样，`prefetch_adaptive_sequential` 现在可以直接回答：

> “我退出之后，UVM BPF policy 到底还在不在？”

#### C. `attention_aware_eviction.c`

除了同样加入 startup / shutdown 的 UVM struct_ops 校验外，还额外加入了：

1. `verify_maps_unpinned("startup")`
2. `verify_maps_unpinned("shutdown")`

用于确认以下 pinned maps 是否真的消失：

- `/sys/fs/bpf/attention_score_map`
- `/sys/fs/bpf/attention_stats_map`

这点对 attention-aware 尤其重要，因为它不仅有 struct_ops，还依赖 pinned map 与 `score_bridge.py` 通信。

### 7.4 改动后的预期行为

现在如果残留没清掉，loader 不会再“默默继续”，而会直接报这种类型的信息：

```text
[startup] Found N UVM struct_ops instance(s):
  - id=...
  - name=...
Refusing to load ...
```

退出时如果没卸干净，也会报：

```text
[shutdown] Found N UVM struct_ops instance(s):
  - id=...
  - name=...
```

attention-aware 还会补充：

```text
[shutdown] score_map pin still exists: /sys/fs/bpf/attention_score_map
[shutdown] stats_map pin still exists: /sys/fs/bpf/attention_stats_map
```

这正是你当前缺少的“运行时真相”。

---

## 8. 回答 `问题.txt` 中的各项问题

### 问题 1：每次重启后重载自定义 575 驱动是否相关？

相关，而且非常关键。

原因有两层：

1. 你在重启后重新 `rmmod/insmod` 自定义 `nvidia.ko` / `nvidia-uvm.ko`，这会重置驱动层的 BPF struct_ops 支持环境。
2. 也正因为这里是“驱动层状态重置点”，所以**第一次运行的实验状态特别重要**。之后如果不彻底清理，后续 run 就可能不是你以为的那个配置。

### 问题 2：第一次纯 UVM 很差、之后再挂 eBPF 仍无改善，是否合理？

从“算法本身”角度看，不合理。

如果 eBPF policy 真的成功接管了 UVM prefetch/eviction 路径，那么它至少不该只带来 2% 左右改善，而另一次又带来 40%+ 改善。

从“实验状态切换失败”角度看，这反而是合理的。

我更倾向于把它解读为：

- 当时“挂载后的状态”并没有被可靠验证；
- 因此你看到的是“以为切换了配置，实际上没有”的结果。

### 问题 3：为什么第一次先挂 eBPF 后，后续卸载再跑纯 UVM 仍保持改善？

这是当前最关键的异常点。

**最强解释**：eBPF 没有被完整卸载，后续“纯 UVM”其实仍然带着已挂载 policy。

这也是本次代码改动重点解决的问题。

### 问题 4：`serve_bench.py` 是否有问题？

结论：**没有发现会伪造这组结论的统计 bug**。

它的问题不是“统计错了”，而是“没有记录当前 BPF 状态”。

### 问题 5：为什么要在两个 extension 程序里加运行时判断？

因为现在缺的不是更多 benchmark，而是**状态证明**。

在 `bpftool` 不可用时，最可靠的办法就是让 loader 自己：

- 启动前确认系统干净；
- 退出后确认已完全卸载；
- 一旦失败就返回非 0 并打印残留对象。

### 问题 6：为什么本次没有重跑命令？

因为你已经明确说明这些命令耗时很长，且不要在排查时重跑。我遵循了这个约束。

### 问题 7：完整详尽文档

本文就是完整文档，包含：

- 对 benchmark 记录程序的检查；
- 对结果的机制解释；
- 对不完整卸载问题的定位；
- 对代码改动的说明；
- 对后续如何判定“真 baseline / 假 baseline”的方法。

---

## 9. 后续建议

### 9.1 现在就可以做的验证

下次实验时，优先观察 loader 输出，而不是只看 benchmark JSON。

你需要的不是“程序能跑完”，而是看到这类明确句子：

- `[startup] Verified: no UVM struct_ops instances remain.`
- `[shutdown] Verified: no UVM struct_ops instances remain.`
- attention-aware 还要看到：
  - `[shutdown] Verified: no pinned attention-aware maps remain.`

如果没有这些句子，就不要把那次结果当成严格 baseline。

### 9.2 建议但本次未改的地方

如果后面希望进一步减少误判，我建议再做两件事：

1. 在 `serve_bench.py` 输出 JSON 时附带一次“当前 BPF 状态快照”。
2. 每次 benchmark 把 loader 的 stdout/stderr 一起保存到对应结果目录中。

这样以后单看结果目录就能知道：

- benchmark 数字是多少；
- 当时到底有没有残留 struct_ops；
- 注意力打分 map 是否已清干净。

---

## 10. 最终判断

基于当前源码和数据，我的最终判断是：

1. **`serve_bench.py` 不是主要问题。**
2. **当前异常结果主要不是算法现象，而是实验状态管理问题。**
3. **最值得怀疑的是 eBPF struct_ops 卸载不完整，导致“无 eBPF baseline”被污染。**
4. **本次改动已经把这个问题变成运行时可观测问题**：以后 loader 会在启动前和退出后明确告诉你，系统里是否还残留 UVM policy。

如果后续你愿意，我们下一步最合适的工作不是盲目重跑大实验，而是先做一轮**短路径状态验证**：

- 只看 loader 日志；
- 不跑完整 100 prompts；
- 先证明 attach / detach / cleanup 的状态转换完全正确；
- 然后再恢复长 benchmark。
