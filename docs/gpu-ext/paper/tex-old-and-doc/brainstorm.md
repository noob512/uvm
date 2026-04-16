% Core new contri:

% story:
% many memory, sched,
% no OS level


% paper:

% hXDP (offload to device)
% XRP (kernel driver)
% cache\_ext/sched\_ext

% (safe, efficient) fine-grain, unified(cross device) and global(multi-tenant), transparent

% 1. support for xxx memory management (pretech, eviction)
%     a. find a interface, where we need hooks 
%     b. build eBPF runtime around it on GPU device for: expanding eBPF into SIMT space
%     c. build a SIMT aware verification for safety
% 2. xxx schedule (GPU: fine-grain thread block level scheduler; driver: addimisson, set priority and premetion)
%     a. find a interface, where we need hooks
%     b. safety
% 3. trace/observebility?

% what we have done

% tracing

% 1. A set bcc style of observability tools on device (gpu memory access, fine-grain scheduler thread block / thread enter / end) neutrino can do
% 2. bcc style trace tools that can show page fault, prefecth decistion / eviction decisitons, GPU scheduler create queue / descrtoy queue / (our interfcae/hooks)
% 3. trace across CPU and GPU (launchlate, etc)

% policy

% 4. memory policy LFU/FIFO/MRU... (driver) preftech policy seq / tree / stide (driver, global)
% 5. GPU side: CLC style work stealing policy, prefetch policy (prefecth some memory without block computing)
% 6. together: prefetch

% eval:
% efficient,
% multi-tenant,
% transparent,
% unified(cross device)

% 1. sync or small kernels (gemm, vector add)
% 2. pytorch / llama.cpp / vllm / faiss
% 3. run multi together



**RQ1（单租户：可编程 memory / scheduling 政策的收益）：在真实 workload 上，gBPF-based policy 比 UVM 默认 + framework 自己玩的 offloading/scheduler 能好多少？**

   * LLM inference（llama.cpp / vLLM）、GNN training、Faiss 向量搜索这些，你 draft 里已经有了初步结果（5×、2×那些）。
   * 但需要更系统：sweep oversub ratio / QPS / context length，而不是只给几个点。



**RQ2（多租户：cross-tenant 视角的价值）：在多租户场景下，gBPF 的“driver+device 全局视角”能在 tail latency、throughput、公平性上做到用户态做不到的事吗？**

   * 这是你 paper 真正能和 Paella / XSched / PILOT / Salus 那堆 work 扯平乃至超一头的点。



**RQ3（可编程性 & 泛用性）：gBPF 真的像 XDP / sched_ext 那样，是一个窄腰，而不是“我给 MoE/LLM hack 了一堆 heuristics”？**

   * 同一套机制是否能表达多种 policy（memory placement / scheduling / observability），对 PyTorch/vLLM/llama.cpp/Faiss 这些不改代码就有效？

你现有的 RQ1–RQ5 完全可以映射到这 4 组，只是需要重新包装一下，让 evaluation 结构围着这 4 组打。


**RQ4（机制成本）：gBPF 作为一个“OS 级 policy substrate”，本身 overhead / 可扩展性是否 acceptable？**

   * driver hook + SIMT-aware verifier + device eBPF runtime + hierarchical map，runtime 自身的延迟/带宽/CPU/GPU 开销是多少，相比 CUPTI/NVBit/eGPU 级别的东西？
