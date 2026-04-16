1.  <a href="https://arxiv.org/html/2512.24637v2#S1" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">1 </span>Introduction</span></a>
2.  <a href="https://arxiv.org/html/2512.24637v2#S2" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">2 </span>Background</span></a>
    1.  <a href="https://arxiv.org/html/2512.24637v2#S2.SS1" class="ltx_ref" title="In Background ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">2.1 </span>Characterizing GPU Tasks</span></a>
    2.  <a href="https://arxiv.org/html/2512.24637v2#S2.SS2" class="ltx_ref" title="In Background ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">2.2 </span>Scheduling GPU Tasks</span></a>
    3.  <a href="https://arxiv.org/html/2512.24637v2#S2.SS3" class="ltx_ref" title="In Background ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">2.3 </span>GPU Memory Multiplexing</span></a>
3.  <a href="https://arxiv.org/html/2512.24637v2#S3" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">3 </span>Rethinking GPU Context Switching</span></a>
4.  <a href="https://arxiv.org/html/2512.24637v2#S4" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">4 </span><span class="ltx_text">MSched</span>Overview</span></a>
    1.  <a href="https://arxiv.org/html/2512.24637v2#S4.SS1" class="ltx_ref" title="In MSchedOverview ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">4.1 </span>Architecture and Workflow</span></a>
    2.  <a href="https://arxiv.org/html/2512.24637v2#S4.SS2" class="ltx_ref" title="In MSchedOverview ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">4.2 </span>Challenges for Proactive Memory Scheduling</span></a>
5.  <a href="https://arxiv.org/html/2512.24637v2#S5" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">5 </span>Memory Access Prediction</span></a>
    1.  <a href="https://arxiv.org/html/2512.24637v2#S5.SS1" class="ltx_ref" title="In Memory Access Prediction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">5.1 </span>Naive Solution: Allocation-granularity Prediction</span></a>
    2.  <a href="https://arxiv.org/html/2512.24637v2#S5.SS2" class="ltx_ref" title="In Memory Access Prediction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">5.2 </span>Our Approach: Template-based Prediction</span></a>
6.  <a href="https://arxiv.org/html/2512.24637v2#S6" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">6 </span>Proactive Memory Scheduling</span></a>
    1.  <a href="https://arxiv.org/html/2512.24637v2#S6.SS1" class="ltx_ref" title="In Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">6.1 </span>Task Scheduler</span></a>
    2.  <a href="https://arxiv.org/html/2512.24637v2#S6.SS2" class="ltx_ref" title="In Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">6.2 </span>Memory Manager</span></a>
    3.  <a href="https://arxiv.org/html/2512.24637v2#S6.SS3" class="ltx_ref" title="In Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">6.3 </span>Page Migration Pipeline</span></a>
7.  <a href="https://arxiv.org/html/2512.24637v2#S7" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">7 </span>Evaluation</span></a>
    1.  <a href="https://arxiv.org/html/2512.24637v2#S7.SS1" class="ltx_ref" title="In Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">7.1 </span>Effectiveness of Proactive Memory Scheduling</span></a>
    2.  <a href="https://arxiv.org/html/2512.24637v2#S7.SS2" class="ltx_ref" title="In Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">7.2 </span>End-to-end Application Performance</span></a>
    3.  <a href="https://arxiv.org/html/2512.24637v2#S7.SS3" class="ltx_ref" title="In Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">7.3 </span>Ablation Study</span></a>
    4.  <a href="https://arxiv.org/html/2512.24637v2#S7.SS4" class="ltx_ref" title="In Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">7.4 </span>Overhead Analysis</span></a>
    5.  <a href="https://arxiv.org/html/2512.24637v2#S7.SS5" class="ltx_ref" title="In Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">7.5 </span>Comparison with Existing Systems</span></a>
8.  <a href="https://arxiv.org/html/2512.24637v2#S8" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">8 </span>Related Work</span></a>
9.  <a href="https://arxiv.org/html/2512.24637v2#S9" class="ltx_ref" title="In Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_title"><span class="ltx_tag ltx_tag_ref">9 </span>Conclusion</span></a>

<div class="ltx_page_main">

<div class="ltx_page_content">

# Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling

<div class="ltx_authors">

<span class="ltx_creator ltx_role_author"> <span class="ltx_personname">  
Weihang Shen, Yinqiu Chen, Rong Chen, Haibo Chen  
Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University  
</span><span class="ltx_author_notes">Work done while Yinqiu Chen was at Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University.Rong Chen is the corresponding author (<a href="https://arxiv.org/html/2512.24637v2/rongchen@sjtu.edu.cn" class="ltx_ref ltx_url ltx_font_typewriter">rongchen@sjtu.edu.cn</a>).</span></span>

</div>

<div class="ltx_abstract">

###### Abstract

The limited HBM capacity has become the primary bottleneck for hosting an increasing number of larger-scale GPU tasks. While demand paging extends capacity via host DRAM, it incurs up to 78$\times$ slowdown due to the massive working sets and poor locality of GPU workloads. We observe, however, that GPU memory access patterns are inherently predictable via kernel launch arguments and their asynchronous execution nature. Leveraging this, we propose <span id="id3.3.1" class="ltx_text">MSched</span>, an OS-level scheduler that extends GPU context switching to include proactive working set preparation, thereby coalescing fragmented, eventual, and expensive page faults into a single efficient migration. <span id="id3.3.2" class="ltx_text">MSched</span> employs a template-based approach to predict working sets with near-perfect accuracy and proposes a co-design between task scheduler and memory manager to enforce a globally optimal page placement policy. Evaluation demonstrates that <span id="id3.3.3" class="ltx_text">MSched</span> outperforms demand paging by up to 11.05$\times$ for scientific and deep learning workloads, and 57.88$\times$ for LLM under memory oversubscription.

</div>

<div id="S1" class="section ltx_section">

## Introduction

<div id="S1.p1" class="ltx_para">

Graphics Processing Units (GPUs) have emerged as the default compute substrate across cloud \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">aegaeon-sosp25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">blitzscale-osdi25</span>\], edge \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">edgenn-icde23</span>\], and personal devices \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">lohan-arxiv24</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">spinfer-eurosys25</span>\]. A growing range of computations are now routinely offloaded to GPUs, including media processing \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">multimedia-tcc20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">clij-naturemethods20</span>\], analytics \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">vectorsearch-fast25</span>\], databases \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">gpudb-vldb23</span>\], and machine learning \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">tgs-nsdi23</span>\]. This widespread adoption creates a pressing demand for running multiple applications concurrently on a single GPU \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">towards-arxiv25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gpu-mt-survey-tpds22</span>\]. For instance, on personal devices like AI PCs, users might be editing presentations with GPU-powered text completion and image generation in foreground, while AI agents and file indexing for RAG service run in background. Due to limited hardware resources, these applications must concurrently execute on the sole GPU. In the cloud, providers also seek to colocate multiple jobs on one GPU \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">tgs-nsdi23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">pipeswitch-osdi20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">shepherd-nsdi23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">paella-sosp23</span>\] to maximize utilization and save costs.

</div>

<div id="S1.p2" class="ltx_para">

Prior research on GPU multitasking \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gpreempt-atc25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">effisha-ppopp17</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">flep-asplos17</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">tally-asplos25</span>\] has primarily focused on compute scheduling, developing various techniques to multiplex GPU computation resources among tasks. These systems implicitly assume that the GPU’s high-bandwidth memory (HBM) can accommodate all concurrent applications. However, an increasing number of applications today become AI-powered and rely on GPU acceleration \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">aiservice-apsys25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">smartapp-arxiv23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">aiui-arxiv24</span>\]. Meanwhile, their memory footprints also escalate rapidly, driven by exponentially growing model sizes and high-resolution, multi-modal inputs. This assumption of sufficient memory no longer holds in most cases. Over the past decade, the HBM capacity of GPUs in each class has only increased by several to tens of gigabytes \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-gpu-list</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">amd-gpu-list</span>\], thereby becoming the primary bottleneck for hosting an increasing number of larger-scale GPU applications concurrently.

</div>

<div id="S1.p3" class="ltx_para">

To handle GPU memory oversubscription, the conventional wisdom is to enable OS demand paging (e.g., Unified Memory), spilling data to larger but slower backing storage such as host CPU DRAM or SSDs \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-um</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">hip-um</span>\]. When the system performs GPU context switching, it lazily swaps only the minimal execution state (e.g., registers and on-chip shared memory) while deferring the working set transition. Memory swapping happens when pages are actually accessed, using GPU virtual memory and page faults to <span id="S1.p3.1.1" class="ltx_text ltx_font_bold">passively</span> migrate pages into HBM via interconnects like PCIe or NVLink C2C \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvlinkc2c</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">nvlinkc2c-benchmark</span>\].

</div>

<figure id="S1.F1.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div id="S1.F1.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<img src="x1.png" id="S1.F1.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="312" alt="Refer to caption" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 1: </span><em>Comparison of total decoding throughput of multiple Llama3-8B (int8-quantized, 8.5 GB each) inference tasks using llama.cpp [<span class="ltx_ref ltx_missing_citation ltx_ref_self">llamacpp</span>] on an NVIDIA RTX 5080 GPU (16 GB HBM) between demand paging (UM [<span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-um</span>]) and <span id="S1.F1.fig1.3.1.1" class="ltx_text ltx_font_upright">MSched</span>.</em></figcaption>
</figure>

<div id="S1.p4" class="ltx_para">

This lazy and passive mechanism is well-suited for traditional CPU workloads, which typically exhibit random and unpredictable memory access patterns but have high temporal locality and small working sets, thereby incurring an acceptable overhead of occasional page faults. GPUs, however, violate these premises. GPU applications often have poor locality and massive memory footprints, accessing gigabytes of data within short execution bursts \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">forest-isca25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">earlyadaptor-ispass23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">etc-asplos19</span>\]. Moreover, GPU page faults are far more expensive than on CPUs: a fault locks the TLB of the GPU compute unit (CU) \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">deepum-asplos23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-um-opt</span>\], stalling thousands of concurrent threads on that CU, and requires CPU intervention to resolve the fault. Our experiments in Fig. <a href="https://arxiv.org/html/2512.24637v2#S1.F1.fig1" class="ltx_ref" title="Figure 1 ‣ Introduction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">1</span></a> quantify this pathology: simply applying demand paging to manage oversubscribed memory for concurrent applications (LLM inference) precipitates a staggering 78$\times$ slowdown. This collapse stems from severe page thrashing, which induces an average of 9,210 page faults per decoding step (12.7 ms). This suggests that this paradigm is ill-suited for supporting memory-demanding GPU multitasking.

</div>

<div id="S1.p5" class="ltx_para ltx_noindent">

<span id="S1.p5.1.1" class="ltx_text ltx_font_bold">Opportunity.</span> We observe that GPU applications have highly predictable memory access patterns, unlike CPU programs, which behave as black boxes to the OS. Most GPU kernels operate on data regions explicitly defined by their launch arguments (e.g., pointers, dimensions, and strides). Moreover, GPU kernels are launched asynchronously in a straightforward sequence, making their complete execution order visible to the OS beforehand. This creates a natural opportunity to predict future memory accesses prior to execution.

</div>

<div id="S1.p6" class="ltx_para ltx_noindent">

<span id="S1.p6.1.1" class="ltx_text ltx_font_bold">Key insight.</span> This predictability motivates us to rethink the very definition of context switching for GPU multitasking. We argue that the notion of GPU “context” should expand beyond the minimal on-chip execution state to include its working set memory. As GPU multitasking workloads often have working sets that exceed HBM capacity, memory itself should also be <span id="S1.p6.1.2" class="ltx_text ltx_font_bold">proactively</span> “scheduled”. In other words, instead of passively handling eventual and expensive page faults during execution, the system should eagerly and proactively prepare the future working set memory for the upcoming task as an integral part of the context switch routine.

</div>

<div id="S1.p7" class="ltx_para ltx_noindent">

<span id="S1.p7.1.1" class="ltx_text ltx_font_bold">Our approach.</span> We introduce <span id="S1.p7.1.2" class="ltx_text">MSched</span>, the first OS-level scheduling system designed for GPU memory sharing among multiple concurrent tasks. <span id="S1.p7.1.3" class="ltx_text">MSched</span> extends GPU memory with host DRAM and treats working set memory of a GPU task as a first-class citizen of its execution context. <span id="S1.p7.1.4" class="ltx_text">MSched</span> achieves this through two core mechanisms. First, it intercepts GPU kernel arguments to build an online forecast of each task’s future memory accesses. Second, <span id="S1.p7.1.5" class="ltx_text">MSched</span> uses this prediction to migrate the corresponding memory pages of the upcoming task into GPU HBM during context switching. This approach coalesces expensive, fragmented page faults into efficient, batched memory transfers, thereby significantly reducing the overhead of memory oversubscription.

</div>

<div id="S1.p8" class="ltx_para ltx_noindent">

<span id="S1.p8.1.1" class="ltx_text ltx_font_bold">Challenges.</span> Realizing this vision of extended context switch, however, is non-trivial. The key challenge lies in ensuring high prediction accuracy in both <span id="S1.p8.1.2" class="ltx_text ltx_framed ltx_framed_underline">spatial</span> and <span id="S1.p8.1.3" class="ltx_text ltx_framed ltx_framed_underline">temporal</span> aspects. First, the prediction must be precise in determining the <span id="S1.p8.1.4" class="ltx_text ltx_framed ltx_framed_underline">spatial</span> regions of future memory accesses: over-prediction wastes valuable bandwidth on unnecessary data, while under-prediction triggers costly page faults that nullify the benefits of proactive memory scheduling. Second, according to Belady’s algorithm \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">opt-1966</span>\], the <span id="S1.p8.1.5" class="ltx_text ltx_framed ltx_framed_underline">temporal</span> sequence of memory accesses directly determines the optimal page placement policy, which is the cornerstone of efficient memory scheduling. However, in multitasking scenarios, frequent context switches cause memory accesses from multiple concurrent tasks to interleave, making it difficult for the system to infer the exact timing of each access and thereby reduce data migration.

</div>

<div id="S1.p9" class="ltx_para">

<span id="S1.p9.1.1" class="ltx_text">MSched</span>overcomes these challenges with two key mechanisms: *template-based memory prediction* and a *scheduler-memory co-design*. To achieve <span id="S1.p9.1.4" class="ltx_text ltx_framed ltx_framed_underline">spatial</span> accuracy, we distill the memory access behaviors of modern GPU workloads into three fundamental templates. <span id="S1.p9.1.5" class="ltx_text">MSched</span> automatically infers the mapping between kernel arguments and accessed memory regions based on these templates, allowing it to precisely (0.25% false negative and 0.00% false positive) predict the working set of each kernel using online argument values. To resolve <span id="S1.p9.1.6" class="ltx_text ltx_framed ltx_framed_underline">temporal</span> interleaving, <span id="S1.p9.1.7" class="ltx_text">MSched</span> exposes the scheduler’s task timeline—a common scheduling primitive that describes future task order and timeslice allocation—to the memory manager. This enables the system to reconstruct a global memory access sequence, thereby enforcing optimal page placement policy aligned with the scheduler’s decisions.

</div>

<div id="S1.p10" class="ltx_para">

We evaluate <span id="S1.p10.10.1" class="ltx_text">MSched</span> under multitasking memory oversubscription. For scientific computing and deep learning workloads, <span id="S1.p10.10.2" class="ltx_text">MSched</span> improves total throughput by 11.05$\times$, 9.35$\times$, and 7.52$\times$ over the native demand paging under 150%, 200%, and 300% memory subscription, respectively. These speedups are magnified for memory-intensive LLM inference, reaching 57.88$\times$, 44.79$\times$, and 33.60$\times$, effectively sustaining 74.09%, 58.23%, and 43.01% of in-HBM performance without oversubscription. Furthermore, <span id="S1.p10.10.3" class="ltx_text">MSched</span> outperforms SUV \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">suv-micro24</span>\], a demand paging optimization for single-task execution, by 7.18$\times$ under 300% memory subscription. Compared to compute-only scheduling (XSched \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\]), <span id="S1.p10.10.4" class="ltx_text">MSched</span> reduces the $P_{99}$ latency of real-time tasks by 4.06$\times$ while boosting the throughput of colocated best-effort tasks by 2.43$\times$. We plan to open-source <span id="S1.p10.10.5" class="ltx_text">MSched</span> to facilitate further development.

</div>

</div>

<div id="S2" class="section ltx_section">

## Background

<div id="S2.SS1" class="section ltx_subsection">

### Characterizing GPU Tasks

<div id="S2.SS1.p1" class="ltx_para">

Unlike general-purpose CPU workloads, which often exhibit complex and branching control flows, GPU tasks follow a structured, host-driven execution model. Typically, the host program allocates buffers in GPU memory to hold data such as inputs, outputs, model parameters, and intermediate results. It then transfers input data into these buffers via memory copy commands. Next, the host launches a series of GPU kernels to perform algorithmic computations on the data. Finally, the processed results are copied back to the host. From the OS perspective, a GPU task simply behaves as a sequence of memory copy and GPU kernel commands \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\].

</div>

<div id="S2.SS1.p2" class="ltx_para">

A distinct feature of this model is the decoupling of command launch and execution. Instead of executing immediately, commands launched by the host CPU are pushed into a software-managed FIFO queue, allowing the host thread to proceed launching the next command. The GPU’s on-chip command processor, e.g., NVIDIA GPU System Processor (GSP) \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-gsp</span>\], then fetches and executes these commands from the queue in strict sequence. This <span id="S2.SS1.p2.1.1" class="ltx_text ltx_font_italic">asynchronous</span> nature exposes a window of <span id="S2.SS1.p2.1.2" class="ltx_text ltx_font_bold">future</span> commands—queued but not yet executed—offering the OS a unique opportunity to proactively analyze and predict the resource requirements of upcoming commands before they actually run on the GPU.

</div>

</div>

<div id="S2.SS2" class="section ltx_subsection">

### Scheduling GPU Tasks

<div id="S2.SS2.p1" class="ltx_para">

To support multitasking, modern GPUs employ time-sharing scheduling to multiplex hardware resources among concurrent GPU tasks. This approach has become the mainstream paradigm in GPU design and deployment due to its superior flexibility and robust fault and performance isolation \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">effisha-ppopp17</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">flep-asplos17</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">tally-asplos25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gpreempt-atc25</span>\]. In this model, the GPU acts as a preemptible resource, arbitrating execution among multiple tasks by performing context switching.

</div>

<div id="S2.SS2.p2" class="ltx_para ltx_noindent">

<span id="S2.SS2.p2.1.1" class="ltx_text ltx_font_bold">GPU context switching.</span> Analogous to CPUs, a GPU execution context is traditionally defined as the *minimal architectural states* resident on the compute units (CUs) that must be saved and restored to ensure correct execution continuation. This includes register files, on-chip shared memory (scratchpad), stack pointers, GPU page table bases, and control flow states (e.g., program counters, active masks, barrier states), with a total size of approximately 200KB per CU. Modern GPUs, such as NVIDIA GPUs after Pascal architecture and Intel Xe GPUs, have natively supported preemption via context switching \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-cilp</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">intel-gpu-specs</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">intel-gpu-context</span>\] to schedule multiple tasks. When preemption is requested (e.g., time slice expiration), the GPU command processor broadcasts an interrupt to the CUs, immediately halting kernel execution and trapping into a software handler. The handler saves the context to GPU HBM and terminates the current kernel. The kernel of the subsequent task is then scheduled onto the CUs to restore its context and resume execution, thereby completing a typical GPU context switching operation. This mechanism has been adopted by state-of-the-art GPU multitasking systems \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gpreempt-atc25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gcaps-ecrts24</span>\] to enable flexible and preemptive GPU scheduling.

</div>

</div>

<div id="S2.SS3" class="section ltx_subsection">

### GPU Memory Multiplexing

<div id="S2.SS3.p1" class="ltx_para ltx_noindent">

<span id="S2.SS3.p1.1.1" class="ltx_text ltx_font_bold">The untenable assumption of sufficient memory.</span> While existing systems \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">tally-asplos25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gpreempt-atc25</span>\] have effectively scheduled GPU compute resources across multiple tasks, they largely overlook the multiplexing of memory resources \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">towards-arxiv25</span>\]. Most scheduling systems operate under an implicit assumption of sufficient memory: they assume that the aggregate memory footprint of all concurrent tasks can simultaneously reside in the GPU HBM. Under this premise, context switching merely involves saving and restoring the minimal architectural states (as described in §<a href="https://arxiv.org/html/2512.24637v2#S2.SS2" class="ltx_ref" title="Scheduling GPU Tasks ‣ Background ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2.2</span></a>), enabling the upcoming task to execute seamlessly as if it held exclusive access to the GPU.

</div>

<div id="S2.SS3.p2" class="ltx_para">

However, this assumption is becoming increasingly untenable. On one hand, the ubiquity of AI-powered applications—ranging from background system agents to foreground interactive tools—has led to a surge in the number of tasks contending for the GPU \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">aiservice-apsys25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">smartapp-arxiv23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">aiui-arxiv24</span>\]. On the other hand, the memory footprint of individual tasks is exploding. Modern GPU workloads, particularly large language models (LLMs), demand massive memory allocations not only for model parameters but also for intermediates and caches. Consequently, the total memory demand of concurrent tasks frequently exceeds the HBM capacity of even high-end GPUs, creating a barrier that simple compute scheduling cannot scale beyond.

</div>

<div id="S2.SS3.p3" class="ltx_para ltx_noindent">

<span id="S2.SS3.p3.1.1" class="ltx_text ltx_font_bold">Demand paging as a workaround.</span> To share the insufficient HBM across these memory-hungry tasks, the conventional solution \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">tgs-nsdi23</span>\] is to enable demand paging for GPUs, such as CUDA Unified Memory (UM) \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-um</span>\] and AMD HIP UM \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">hip-um</span>\]. This mechanism extends the GPU memory space with host CPU DRAM as a cheaper and larger backing storage. In this architecture, the GPU driver maintains a unified virtual address space where the physical pages are dynamically migrated between device HBM and host DRAM. When a running kernel accesses a virtual page that is not resident in HBM, the GPU Memory Management Unit (MMU) raises a page fault, temporarily stalls the execution of the faulting compute unit and signals an interrupt to the CPU. The GPU driver on the CPU catches the interrupt and migrates the faulting page from host DRAM to HBM through high-bandwidth interconnects like PCIe or NVLink C2C \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvlinkc2c</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">nvlinkc2c-benchmark</span>\]. If the HBM is already full, the driver must evict pages to free up space. Once the eviction and migration complete, the driver updates the GPU page table and resumes the stalled compute unit. By transparently and <span id="S2.SS3.p3.1.2" class="ltx_text ltx_font_bold">passively</span> migrating pages upon access, demand paging can be directly integrated into existing scheduling systems and theoretically allows multiple memory-consuming tasks to oversubscribe GPU memory.

</div>

</div>

</div>

<div id="S3" class="section ltx_section">

## Rethinking GPU Context Switching

<div id="S3.p1" class="ltx_para">

While demand paging extends GPU memory beyond its physical HBM capacity, its performance suffers from severe degradation in multitasking scenarios. Based on our experiments in Fig. <a href="https://arxiv.org/html/2512.24637v2#S1.F1.fig1" class="ltx_ref" title="Figure 1 ‣ Introduction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">1</span></a>, simply enabling demand paging (UM) for oversubscribed concurrent LLM inference tasks results in a 78$\times$ slowdown compared to running with sufficient HBM. This pathological degradation compels us to examine why this passive demand paging—effective in the CPU world—fails in the scenario of GPU multitasking. We identify three key limitations as follows.

</div>

<div id="S3.p2" class="ltx_para ltx_noindent">

<span id="S3.p2.2.1" class="ltx_text ltx_font_bold">Lengthy control plane of GPU page faults.</span> Unlike CPUs, GPU page faults are exorbitantly expensive due to hardware constraints. First, as mentioned in §<a href="https://arxiv.org/html/2512.24637v2#S2.SS3" class="ltx_ref" title="GPU Memory Multiplexing ‣ Background ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2.3</span></a>, resolving a GPU page fault necessitates intervention from the driver on the host CPU, entailing multiple interrupt round-trips across the PCIe bus. Our measurements on RTX 5080 GPU (PCIe 5.0) show that handling a single GPU page fault takes 31.79 $\mu\hspace{0pt}s$, of which only 1.35 $\mu\hspace{0pt}s$ is spent on data transfer, while the remaining 96% is spent on control plane. Second, in terms of indirect overhead, a page fault triggered by a single GPU thread locks the TLB of the entire CU and prevents further address translations until the fault is resolved \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">deepum-asplos23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-um-opt</span>\], stalling thousands of concurrent GPU threads on that CU. The massive parallelism of GPUs ironically amplifies the penalty of these stalls, causing severe underutilization of the hardware.

</div>

<figure id="S3.F2.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_center ltx_align_middle" style="width:433.6pt;">
<div id="S3.F2.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_center ltx_align_middle" style="width:429.3pt;">
<img src="x2.png" id="S3.F2.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="372" alt="Refer to caption" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 2: </span><em>GPU memory access pattern (sorted) of Llama3-8B (int8-quantized) inference with llama.cpp on an NVIDIA RTX 5080 GPU.</em></figcaption>
</figure>

<div id="S3.p3" class="ltx_para ltx_noindent">

<span id="S3.p3.1.1" class="ltx_text ltx_font_bold">Poor locality and large working set.</span> Demand paging is specifically designed for workloads with *strong temporal locality* and *small active working sets* that rarely exceed the physical memory capacity. However, GPU applications often violate these premises. As illustrated in Fig. <a href="https://arxiv.org/html/2512.24637v2#S3.F2.fig1" class="ltx_ref" title="Figure 2 ‣ Rethinking GPU Context Switching ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2</span></a>, typical GPU workloads (e.g., LLM inference) frequently touch massive data, essentially their entire memory allocation (8.5 GB), within short periods (12.7 ms). Such streaming-like access pattern exhibits *poor temporal locality* and *large working sets* whose aggregate size across concurrent tasks easily surpasses HBM capacity. Consequently, demand paging triggers storms of page fault and severe memory thrashing, amplifying the fault handling overhead to a prohibitive level.

</div>

<div id="S3.p4" class="ltx_para ltx_noindent">

<span id="S3.p4.1.1" class="ltx_text ltx_font_bold">Scheduling-induced task switching.</span> Demand paging passively ignores the complete working set shift introduced by context switching in multitasking scenarios. As a result, the incoming task is forced to fault in its working set page-by-page and cold start after every timeslice. Moreover, existing optimizations for demand paging \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">etc-asplos19</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">swapadvisor-asplos20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">deepum-asplos23</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">suv-micro24</span>\] focus solely on intra-task memory management while ignoring the impact of inter-task scheduling and context switches on memory behavior. This oversight leads to severe page migration conflicts between tasks, preventing them from effectively scaling to multitasking workloads.

</div>

<div id="S3.p5" class="ltx_para ltx_noindent">

<span id="S3.p5.1.1" class="ltx_text ltx_font_bold">Opportunity.</span> GPU workloads have a fundamental characteristic: their memory access behaviors are inherently predictable and explicitly exposed to the OS. First, GPU kernel launch arguments specify the memory buffers that the kernel will read from or write to. For example, a matrix multiplication kernel <span id="S3.p5.1.2" class="ltx_text ltx_font_typewriter">matmul(A, B, C,…)</span> typically reads and multiplies the matrices stored in pointers <span id="S3.p5.1.3" class="ltx_text ltx_font_typewriter">A</span> and <span id="S3.p5.1.4" class="ltx_text ltx_font_typewriter">B</span>, writing the result to pointer <span id="S3.p5.1.5" class="ltx_text ltx_font_typewriter">C</span>. Second, as mentioned in §<a href="https://arxiv.org/html/2512.24637v2#S2.SS1" class="ltx_ref" title="Characterizing GPU Tasks ‣ Background ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2.1</span></a>, a GPU task is composed of a well-defined sequence of kernels launched asynchronously, exposing a deterministic order of future kernel execution. This predictability empowers the OS to forecast the working set of GPU tasks ahead of execution.

</div>

<div id="S3.p6" class="ltx_para ltx_noindent">

<span id="S3.p6.2.1" class="ltx_text ltx_font_bold">Extending GPU Context Switching.</span> The predictability nature substantially changes the situation of GPU scheduling: it eliminates the uncertainty of memory access that compels CPUs to rely on demand paging. We argue that GPU memory management should no longer be a passive fault-handling manner but a <span id="S3.p6.2.2" class="ltx_text ltx_font_bold">proactive</span> scheduling paradigm. Specifically, we propose to expand the definition of a GPU context switch to include the <span id="S3.p6.2.3" class="ltx_text ltx_font_italic">proactive restoration of working set memory</span>. Instead of relying on page faults to trigger memory migration, the system should leverage the predictability to identify the memory required by the upcoming task and eagerly preload it into HBM before the task resumes execution. The benefits of this paradigm shift are twofold. First, it eliminates the control-plane overhead of handling massive page faults and the indirect cost from blocking thousands of other concurrent GPU threads. Second, it batches fragmented and page-by-page data movement into one complete and efficient operation. Our evaluation on an NVIDIA RTX 5080 GPU with PCIe 5.0$\times$<!-- -->16 interconnect highlights this disparity: while fine-grained migration via page faults yields a meager effective throughput of 0.12 GB/s, batched transfer saturates the interconnect at 41.7 GB/s—a 347$\times$ improvement in bandwidth efficiency.

</div>

</div>

<div id="S4" class="section ltx_section">

## <span id="S4.1.1" class="ltx_text ltx_font_bold" style="font-size:120%;">MSched</span><span id="S4.2.2" class="ltx_text ltx_font_bold" style="font-size:120%;">Overview</span>

<div id="S4.p1" class="ltx_para">

Building on the idea of proactive memory scheduling and the augmented notion of context switching, we introduce <span id="S4.p1.1.1" class="ltx_text">MSched</span>, the first OS-level scheduler tailored for multitasking GPU workloads under memory oversubscription. <span id="S4.p1.1.2" class="ltx_text">MSched</span> extends GPU memory capacity by spilling inactive memory pages to host CPU DRAM and schedules data movement to keep the active task’s working set resident in GPU HBM. It first identifies the memory accesses of each task based on its kernel launch arguments. Next, unlike conventional demand paging, <span id="S4.p1.1.3" class="ltx_text">MSched</span> proactively migrates memory during context switches: it evicts currently idle pages to host DRAM while simultaneously loading the next task’s working set into GPU HBM, enabling execution to proceed without page fault stalls.

</div>

<figure id="S4.F3.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div id="S4.F3.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<img src="x3.png" id="S4.F3.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="598" height="420" alt="Refer to caption" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 3: </span><em>Architecture and workflow of <span id="S4.F3.fig1.3.1.1" class="ltx_text ltx_font_upright">MSched</span>.</em></figcaption>
</figure>

<div id="S4.SS1" class="section ltx_subsection">

### Architecture and Workflow

<div id="S4.SS1.p1" class="ltx_para">

Fig. <a href="https://arxiv.org/html/2512.24637v2#S4.F3.fig1" class="ltx_ref" title="Figure 3 ‣ MSchedOverview ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">3</span></a> illustrates the architecture and workflow of <span id="S4.SS1.p1.1.1" class="ltx_text">MSched</span>. It consists of an offline part, which profiles the kernel execution latency and analyzes the argument-memory relationship, and an online part, which predicts and proactively schedules the memory for all running tasks in the system.

</div>

<div id="S4.SS1.p2" class="ltx_para ltx_noindent">

<span id="S4.SS1.p2.1.1" class="ltx_text ltx_font_bold">Task analysis (Offline).</span> During the offline phase, a kernel profiler measures the execution latency of every GPU kernel in user applications. It also intercepts kernel arguments and records the memory addresses touched by each kernel. A memory analyzer then examines these arguments and memory traces to construct a mapping between them based on a set of predefined templates. Subsequently, it generates a description file that encapsulates kernel latencies and formulas for calculating memory regions. This offline phase can be integrated into the compiler or executed during installation.

</div>

<div id="S4.SS1.p3" class="ltx_para ltx_noindent">

<span id="S4.SS1.p3.1.1" class="ltx_text ltx_font_bold">Memory scheduling (Online).</span> The online part of <span id="S4.SS1.p3.1.2" class="ltx_text">MSched</span> comprises four key modules: a predictor, a task scheduler, a memory manager, and a modified GPU driver.

</div>

<div id="S4.SS1.p4" class="ltx_para ltx_noindent">

<span id="S4.SS1.p4.1.1" class="ltx_text ltx_font_italic ltx_framed ltx_framed_underline">Predictor.</span> The predictor is a dynamically linked library (DLL) that is preloaded into each GPU application process. It intercepts kernel launch APIs and their arguments, accurately predicting the memory pages each kernel will access based on the argument-memory mapping generated in the offline phase. The predicted memory access information is then attached to the kernel metadata and passed to the memory manager.

</div>

<div id="S4.SS1.p5" class="ltx_para ltx_noindent">

<span id="S4.SS1.p5.1.1" class="ltx_text ltx_font_italic ltx_framed ltx_framed_underline">Task scheduler (Extended).</span> To schedule GPU tasks, <span id="S4.SS1.p5.1.2" class="ltx_text">MSched</span> extends a GPU scheduler (XSched \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\]) that enables preemptive GPU context switching among multiple tasks. When switching to a new task, the context switcher triggers the memory manager to perform proactive memory scheduling.

</div>

<div id="S4.SS1.p6" class="ltx_para ltx_noindent">

<span id="S4.SS1.p6.1.1" class="ltx_text ltx_font_italic ltx_framed ltx_framed_underline">Memory manager.</span> Invoked by the scheduler upon context switching, the memory manager evicts pages to host CPU DRAM and follows the optimal replacement policy (OPT, i.e., Belady’s algorithm \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">opt-1966</span>\]) to minimize the overall migration volume. This process is guided by the predicted memory pages, profiled kernel latencies, and the scheduler’s task timeline. The manager then accurately populates the working set (wset) of the next task into GPU HBM via <span id="S4.SS1.p6.1.2" class="ltx_text ltx_font_typewriter">ioctl</span> of the GPU driver.

</div>

<div id="S4.SS1.p7" class="ltx_para ltx_noindent">

<span id="S4.SS1.p7.1.1" class="ltx_text ltx_font_italic ltx_framed ltx_framed_underline">GPU driver (Extended).</span> <span id="S4.SS1.p7.1.2" class="ltx_text">MSched</span> modifies the kernel-mode GPU driver and implements a new <span id="S4.SS1.p7.1.3" class="ltx_text ltx_font_typewriter">madvise</span> ioctl to enforce the OPT eviction sequence, and a <span id="S4.SS1.p7.1.4" class="ltx_text ltx_font_typewriter">migrate</span> ioctl to proactively transfer memory between DRAM and HBM. <span id="S4.SS1.p7.1.5" class="ltx_text">MSched</span> exploits the dual DMA engines widely available in modern GPUs \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-guide</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">amd-ce</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">demystify-rtas24</span>\] and the full-duplex bandwidth of PCIe interconnect, enabling parallel page eviction and population.

</div>

<div id="S4.SS1.p8" class="ltx_para ltx_noindent">

<span id="S4.SS1.p8.1.1" class="ltx_text ltx_font_bold">Transparency.</span> It is important to note that <span id="S4.SS1.p8.1.2" class="ltx_text">MSched</span> is implemented at the OS level, providing complete transparency to user applications. It requires neither application modifications nor access to kernel source code, ensuring full compatibility with closed-source platforms such as CUDA.

</div>

</div>

<div id="S4.SS2" class="section ltx_subsection">

### Challenges for Proactive Memory Scheduling

<div id="S4.SS2.p1" class="ltx_para">

Despite the simplicity of the idea, realizing such proactive memory scheduling and extended context switching is challenging, especially in achieving <span id="S4.SS2.p1.1.1" class="ltx_text ltx_font_bold">accurate</span> memory prediction in both <span id="S4.SS2.p1.1.2" class="ltx_text ltx_framed ltx_framed_underline">spatial</span> and <span id="S4.SS2.p1.1.3" class="ltx_text ltx_framed ltx_framed_underline">temporal</span> dimensions. First, in terms of <span id="S4.SS2.p1.1.4" class="ltx_text ltx_framed ltx_framed_underline">spatial</span> accuracy, prediction must be precise in identifying which memory pages will be accessed within the microsecond-scale window between kernel launch and execution. If the prediction fails to cover the entire working set (under-prediction), the system will fall back to inevitable and inefficient page faults. Conversely, over-prediction not only migrates superfluous data but, more critically, evicts pages that should have been retained for other tasks, causing sustained performance degradation in subsequent tasks. Second, the system must infer the exact <span id="S4.SS2.p1.1.5" class="ltx_text ltx_framed ltx_framed_underline">temporal</span> order of future accesses under the interleaving of multiple concurrent tasks, which serves as the basis for the globally optimal page replacement policy (OPT). Although the execution latencies of individual kernels are deterministic \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">clockwork-osdi20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">rammer-osdi20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>\], scheduler decisions significantly reshape the sequence in which memory is consumed. Therefore, optimizing the global migration volume requires aligning eviction and placement with the plan of computation scheduler to avoid policy conflicts. We address these challenges through *template-based memory prediction* and a *co-design between task scheduling and memory management*, as detailed in the following two sections.

</div>

</div>

</div>

<div id="S5" class="section ltx_section">

## Memory Access Prediction

<div id="S5.p1" class="ltx_para">

The efficacy of <span id="S5.p1.1.1" class="ltx_text">MSched</span> hinges on its ability to accurately identify the working set of each GPU task at runtime. The primary objective is to <span id="S5.p1.1.2" class="ltx_text ltx_font_italic">maximize prediction accuracy</span>: minimizing the false negative rate (under-prediction) to prevent expensive page faults, while simultaneously minimizing the false positive rate (over-prediction) to avoid wasting interconnect bandwidth and polluting the limited GPU HBM with unused data. Furthermore, this prediction mechanism must be highly efficient, as the asynchronous time window between command launching and execution is typically microsecond-scale \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">picker-arxiv24</span>\], leaving little budget for per-kernel analysis.

</div>

<div id="S5.p2" class="ltx_para">

A GPU task typically comprises memory copy commands and compute kernels. Predicting the memory footprint of copy commands is straightforward, as their semantics are explicitly defined by the API: the source, destination, and transfer size are provided as direct arguments. However, predicting the memory access of GPU kernels presents a significant challenge. Due to the flexible programmability of GPUs, kernel semantics are opaque to the OS. Although the base addresses of data buffers are passed as kernel arguments, the actual range of memory accessed is implicitly determined by the kernel’s internal logic. Consequently, the crux of the problem lies in precisely inferring the memory access boundaries for arbitrary GPU kernels.

</div>

<div id="S5.SS1" class="section ltx_subsection">

### Naive Solution: Allocation-granularity Prediction

<div id="S5.SS1.p1" class="ltx_para">

Given that any memory region accessed by a kernel must have been allocated via GPU memory management APIs (e.g., <span id="S5.SS1.p1.1.1" class="ltx_text ltx_font_typewriter">cudaMalloc</span>), a naive solution \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">phos-sosp25</span>\] tracks all memory allocation and deallocation requests to maintain a map of valid memory buffers. Upon a kernel launch, the system checks each pointer argument. If a pointer falls within a known allocated buffer, the system conservatively predicts that the kernel will access the entire buffer.

</div>

<div id="S5.SS1.p2" class="ltx_para">

While conceptually simple, this allocation-based approach suffers from severe over-prediction in practice due to two aspects. <span id="S5.SS1.p2.1.1" class="ltx_text ltx_font_bold">Aggregated allocation</span>: applications often consolidate multiple data objects into a single large contiguous buffer. For example, llama.cpp \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">llamacpp</span>\] in Fig. <a href="https://arxiv.org/html/2512.24637v2#S3.F2.fig1" class="ltx_ref" title="Figure 2 ‣ Rethinking GPU Context Switching ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2</span></a> allocates monolithic buffers for the entire model weights (7.6 GB) and activations (0.3 GB), subsequently slicing them for individual layers. Although a specific kernel only accesses a small, layer-specific slice (16 KB–60 MB), the OS perceives the kernel pointer as referencing the entire allocation. Similarly, modern deep learning frameworks like PyTorch \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">pytorch-malloc</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">towards-arxiv25</span>\], TensorFlow \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">tensorflow-malloc</span>\], and JAX \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">jax-malloc</span>\] also pre-allocate massive memory pools (often spanning GBs) to bypass the OS allocator and implement customized sub-allocation internally. <span id="S5.SS1.p2.1.2" class="ltx_text ltx_font_bold">Sparse access</span>: even when a memory buffer is exclusively dedicated to a single data object, the kernel may only touch a fraction of it. A prime example is the KV cache in LLMs: while the buffer is allocated for the maximum context length, the inference kernel only accesses tokens up to the current sequence length (4 KB).

</div>

<div id="S5.SS1.p3" class="ltx_para">

Our empirical analysis in Table <a href="https://arxiv.org/html/2512.24637v2#S5.T1.fig1" class="ltx_ref" title="Table 1 ‣ Naive Solution: Allocation-granularity Prediction ‣ Memory Access Prediction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">1</span></a> confirms the inadequacy of this approach. While the false positive rate is 31% for scientific computing benchmarks, it surges to 80% on average for standard deep learning workloads. Most critically, for LLM inference, the false positive rate reaches an alarming 99.7%. This indicates that most of the memory pages predicted by this naive approach are not actually accessed, leading to gigabytes of unnecessary migration and HBM pollution.

</div>

<figure id="S5.T1.fig1" class="ltx_table ltx_figure_panel ltx_minipage ltx_align_center ltx_align_middle" style="width:433.6pt;">
<br />

<table id="S5.T1.fig1.3" class="ltx_tabular ltx_centering ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<tbody>
<tr id="S5.T1.fig1.3.1" class="odd ltx_tr">
<td rowspan="2" id="S5.T1.fig1.3.1.1" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.1.1.1" class="ltx_text" style="font-size:90%;">Application</span></td>
<td rowspan="2" id="S5.T1.fig1.3.1.2" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.1.2.1" class="ltx_text" style="font-size:90%;">Model</span></td>
<td colspan="2" id="S5.T1.fig1.3.1.3" class="ltx_td ltx_align_center ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.1.3.1" class="ltx_text" style="font-size:90%;">Allocation</span></td>
<td colspan="2" id="S5.T1.fig1.3.1.4" class="ltx_td ltx_align_center ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.1.4.1" class="ltx_text" style="font-size:90%;">Template</span></td>
</tr>
<tr id="S5.T1.fig1.3.2" class="even ltx_tr">
<td id="S5.T1.fig1.3.2.1" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.2.1.1" class="ltx_text" style="font-size:90%;">(F-)</span></td>
<td id="S5.T1.fig1.3.2.2" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.2.2.1" class="ltx_text" style="font-size:90%;">(F+)</span></td>
<td id="S5.T1.fig1.3.2.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.2.3.1" class="ltx_text" style="font-size:90%;">(F-)</span></td>
<td id="S5.T1.fig1.3.2.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.2.4.1" class="ltx_text" style="font-size:90%;">(F+)</span></td>
</tr>
<tr id="S5.T1.fig1.3.3" class="odd ltx_tr">
<td id="S5.T1.fig1.3.3.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.3.1.1" class="ltx_text" style="font-size:90%;">Rodinia</span></td>
<td id="S5.T1.fig1.3.3.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.3.2.1" class="ltx_text" style="font-size:90%;">/</span></td>
<td id="S5.T1.fig1.3.3.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.3.3.1" class="ltx_text" style="font-size:90%;">  0.10</span></td>
<td id="S5.T1.fig1.3.3.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.3.4.1" class="ltx_text" style="font-size:90%;">31.16</span></td>
<td id="S5.T1.fig1.3.3.5" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.3.5.1" class="ltx_text" style="font-size:90%;">  0.92</span></td>
<td id="S5.T1.fig1.3.3.6" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.3.6.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T1.fig1.3.4" class="even ltx_tr">
<td id="S5.T1.fig1.3.4.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.4.1.1" class="ltx_text" style="font-size:90%;">PyTorch/Train</span></td>
<td id="S5.T1.fig1.3.4.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.4.2.1" class="ltx_text" style="font-size:90%;">R</span></td>
<td id="S5.T1.fig1.3.4.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.4.3.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T1.fig1.3.4.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.4.4.1" class="ltx_text" style="font-size:90%;">52.19</span></td>
<td id="S5.T1.fig1.3.4.5" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.4.5.1" class="ltx_text" style="font-size:90%;">  0.52</span></td>
<td id="S5.T1.fig1.3.4.6" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.4.6.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T1.fig1.3.5" class="odd ltx_tr">
<td rowspan="4" id="S5.T1.fig1.3.5.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.5.1.1" class="ltx_text" style="font-size:90%;">PyTorch/Infer</span></td>
<td id="S5.T1.fig1.3.5.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.5.2.1" class="ltx_text" style="font-size:90%;">R</span></td>
<td id="S5.T1.fig1.3.5.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.5.3.1" class="ltx_text" style="font-size:90%;">  0.03</span></td>
<td id="S5.T1.fig1.3.5.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.5.4.1" class="ltx_text" style="font-size:90%;">89.44</span></td>
<td id="S5.T1.fig1.3.5.5" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.5.5.1" class="ltx_text" style="font-size:90%;">  0.17</span></td>
<td id="S5.T1.fig1.3.5.6" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.5.6.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T1.fig1.3.6" class="even ltx_tr">
<td id="S5.T1.fig1.3.6.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.6.1.1" class="ltx_text" style="font-size:90%;">V</span></td>
<td id="S5.T1.fig1.3.6.2" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.6.2.1" class="ltx_text" style="font-size:90%;">  0.03</span></td>
<td id="S5.T1.fig1.3.6.3" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.6.3.1" class="ltx_text" style="font-size:90%;">78.32</span></td>
<td id="S5.T1.fig1.3.6.4" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.6.4.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T1.fig1.3.6.5" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.6.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T1.fig1.3.7" class="odd ltx_tr">
<td id="S5.T1.fig1.3.7.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.7.1.1" class="ltx_text" style="font-size:90%;">I</span></td>
<td id="S5.T1.fig1.3.7.2" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.7.2.1" class="ltx_text" style="font-size:90%;">  0.02</span></td>
<td id="S5.T1.fig1.3.7.3" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.7.3.1" class="ltx_text" style="font-size:90%;">89.32</span></td>
<td id="S5.T1.fig1.3.7.4" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.7.4.1" class="ltx_text" style="font-size:90%;">  0.23</span></td>
<td id="S5.T1.fig1.3.7.5" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.7.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T1.fig1.3.8" class="even ltx_tr">
<td id="S5.T1.fig1.3.8.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.8.1.1" class="ltx_text" style="font-size:90%;">D</span></td>
<td id="S5.T1.fig1.3.8.2" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.8.2.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T1.fig1.3.8.3" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.8.3.1" class="ltx_text" style="font-size:90%;">90.84</span></td>
<td id="S5.T1.fig1.3.8.4" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.8.4.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T1.fig1.3.8.5" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.8.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T1.fig1.3.9" class="odd ltx_tr">
<td id="S5.T1.fig1.3.9.1" class="ltx_td ltx_align_left ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.9.1.1" class="ltx_text" style="font-size:90%;">llama.cpp</span></td>
<td id="S5.T1.fig1.3.9.2" class="ltx_td ltx_align_left ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.9.2.1" class="ltx_text" style="font-size:90%;">L</span></td>
<td id="S5.T1.fig1.3.9.3" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.9.3.1" class="ltx_text" style="font-size:90%;">  0.04</span></td>
<td id="S5.T1.fig1.3.9.4" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.9.4.1" class="ltx_text" style="font-size:90%;">99.70</span></td>
<td id="S5.T1.fig1.3.9.5" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.9.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T1.fig1.3.9.6" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T1.fig1.3.9.6.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
</tbody>
</table>
<br />

<figcaption><span class="ltx_tag ltx_tag_table">Table 1: </span><span id="S5.T1.fig1.2.1" class="ltx_text ltx_font_italic" style="font-size:90%;">Comparison of kernel-level false negative rate <span id="S5.T1.fig1.2.1.1" class="ltx_text ltx_font_upright">(F-)</span> and false positive rate <span id="S5.T1.fig1.2.1.2" class="ltx_text ltx_font_upright">(F+)</span> between two prediction approaches. Models: <span id="S5.T1.fig1.2.1.3" class="ltx_text ltx_font_upright ltx_framed ltx_framed_underline">R</span><span id="S5.T1.fig1.2.1.4" class="ltx_text ltx_font_upright">esNet152, <span id="S5.T1.fig1.2.1.4.1" class="ltx_text ltx_framed ltx_framed_underline">V</span>GG19, <span id="S5.T1.fig1.2.1.4.2" class="ltx_text ltx_framed ltx_framed_underline">I</span>nceptionV3, <span id="S5.T1.fig1.2.1.4.3" class="ltx_text ltx_framed ltx_framed_underline">D</span>enseNet201, <span id="S5.T1.fig1.2.1.4.4" class="ltx_text ltx_framed ltx_framed_underline">L</span>lama3-8B.</span></span></figcaption>
</figure>

</div>

<div id="S5.SS2" class="section ltx_subsection">

### Our Approach: Template-based Prediction

<div id="S5.SS2.p1" class="ltx_para">

We observe that beyond base pointers passed as kernel arguments, the size and structure of a kernel’s memory access region also inherently correlate with its launch arguments, including kernel calling arguments and launch configuration (e.g., grid and thread-block dimensions). Our approach, therefore, analyzes the runtime history of kernel executions to discover the mapping between launch arguments and memory access boundaries, and then encodes this mapping into concise formulas that can be efficiently evaluated online.

</div>

<div id="S5.SS2.p2" class="ltx_para ltx_noindent">

<span id="S5.SS2.p2.1.1" class="ltx_text ltx_font_bold">Kernel profiler.</span> To infer these mappings, we developed an offline kernel profiler based on NVBit \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvbit-micro19</span>\], a binary instrumentation tool for GPU kernels. The profiler instruments every memory access instruction in the target kernel and records the accessed addresses to identify the memory regions the kernel touches. Meanwhile, it intercepts the kernel launch API to capture all corresponding launch arguments. We profile representative GPU workloads spanning scientific computing, deep learning inference and training, and LLMs.

</div>

<div id="S5.SS2.p3" class="ltx_para ltx_noindent">

<span id="S5.SS2.p3.1.1" class="ltx_text ltx_font_bold">Access pattern templates.</span> We analyze the memory traces collected by the profiler and classify the access behaviors of GPU kernels. The results summarized in Table <a href="https://arxiv.org/html/2512.24637v2#S5.T2.fig1" class="ltx_ref" title="Table 2 ‣ Our Approach: Template-based Prediction ‣ Memory Access Prediction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2</span></a> reveal that despite the algorithmic diversity of GPU kernels, their memory access patterns are highly structured. Common to all patterns is that the memory region begins with a <span id="S5.SS2.p3.1.2" class="ltx_text ltx_font_italic">base address</span> provided by a specific pointer argument, while the <span id="S5.SS2.p3.1.3" class="ltx_text ltx_font_italic">spatial extent</span> (i.e., size and shape) intrinsically follows one of the three fundamental templates listed below:

</div>

<div id="S5.SS2.p4" class="ltx_para">

- <span id="S5.I1.i1"><span class="ltx_tag ltx_tag_item">•</span></span>
  <div id="S5.I1.i1.p1" class="ltx_para">

  <span id="S5.I1.i1.p1.1.1" class="ltx_text ltx_font_bold">T1: Fixed-size access</span> ($\sim$<!-- -->77%). The size of the accessed region is fixed across all invocations of the kernel, either independent of the launch arguments, or determined by invariant arguments. This pattern is the most common, since kernels are often invoked in a consistent manner throughout the application’s lifecycle, with most launch arguments remaining unchanged.

  </div>
- <span id="S5.I1.i2"><span class="ltx_tag ltx_tag_item">•</span></span>
  <div id="S5.I1.i2.p1" class="ltx_para">

  <span id="S5.I1.i2.p1.4.4" class="ltx_text ltx_font_bold">T2: Linear-size access</span> ($\sim$<!-- -->18%). The accessed region is contiguous, but its size scales linearly with the product of specific launch arguments. This pattern is prevalent in ML workloads, where the number of elements or tensor dimensions are explicitly specified by arguments. For example, in kernel <span id="S5.I1.i2.p1.4.5" class="ltx_text ltx_font_typewriter">vector_add(A, B, C, N)</span>, the accessed sizes of buffers <span id="S5.I1.i2.p1.4.6" class="ltx_text ltx_font_typewriter">A</span>, <span id="S5.I1.i2.p1.4.7" class="ltx_text ltx_font_typewriter">B</span>, and <span id="S5.I1.i2.p1.4.8" class="ltx_text ltx_font_typewriter">C</span> scale linearly with element count <span id="S5.I1.i2.p1.4.9" class="ltx_text ltx_font_typewriter">N</span>, while in <span id="S5.I1.i2.p1.4.10" class="ltx_text ltx_font_typewriter">matrix_mul(A, B, C, M, N, K)</span>, they scale with dimensions <span id="S5.I1.i2.p1.2.1" class="ltx_text ltx_font_typewriter">M$\times$K</span>, <span id="S5.I1.i2.p1.3.2" class="ltx_text ltx_font_typewriter">K$\times$N</span>, and <span id="S5.I1.i2.p1.4.3" class="ltx_text ltx_font_typewriter">M$\times$N</span>, respectively.

  </div>
- <span id="S5.I1.i3"><span class="ltx_tag ltx_tag_item">•</span></span>
  <div id="S5.I1.i3.p1" class="ltx_para">

  <span id="S5.I1.i3.p1.1.1" class="ltx_text ltx_font_bold">T3: Strided access</span> ($\sim$<!-- -->5%). The access pattern consists of multiple discontiguous memory chunks separated by regular strides, where the stride and chunk size are also linear to the product of specific launch arguments. This pattern typically arises in ML workloads, when kernels operate on specific dimensions of high-dimensional tensors.

  </div>

</div>

<figure id="S5.T2.fig1" class="ltx_table ltx_figure_panel ltx_minipage ltx_align_center ltx_align_middle" style="width:433.6pt;">
<br />

<table id="S5.T2.fig1.3" class="ltx_tabular ltx_centering ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<tbody>
<tr id="S5.T2.fig1.3.1" class="odd ltx_tr">
<td id="S5.T2.fig1.3.1.1" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.1.1.1" class="ltx_text" style="font-size:90%;">Application</span></td>
<td id="S5.T2.fig1.3.1.2" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.1.2.1" class="ltx_text" style="font-size:90%;">Model</span></td>
<td id="S5.T2.fig1.3.1.3" class="ltx_td ltx_align_center ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.1.3.1" class="ltx_text" style="font-size:90%;">Fixed</span></td>
<td id="S5.T2.fig1.3.1.4" class="ltx_td ltx_align_center ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.1.4.1" class="ltx_text" style="font-size:90%;">Linear</span></td>
<td id="S5.T2.fig1.3.1.5" class="ltx_td ltx_align_center ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.1.5.1" class="ltx_text" style="font-size:90%;">Strided</span></td>
<td id="S5.T2.fig1.3.1.6" class="ltx_td ltx_align_center ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.1.6.1" class="ltx_text" style="font-size:90%;">Others</span></td>
</tr>
<tr id="S5.T2.fig1.3.2" class="even ltx_tr">
<td id="S5.T2.fig1.3.2.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.2.1.1" class="ltx_text" style="font-size:90%;">Rodinia</span></td>
<td id="S5.T2.fig1.3.2.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.2.2.1" class="ltx_text" style="font-size:90%;">/</span></td>
<td id="S5.T2.fig1.3.2.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.2.3.1" class="ltx_text" style="font-size:90%;">99.08</span></td>
<td id="S5.T2.fig1.3.2.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.2.4.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T2.fig1.3.2.5" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.2.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
<td id="S5.T2.fig1.3.2.6" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.2.6.1" class="ltx_text" style="font-size:90%;">  0.92</span></td>
</tr>
<tr id="S5.T2.fig1.3.3" class="odd ltx_tr">
<td id="S5.T2.fig1.3.3.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.3.1.1" class="ltx_text" style="font-size:90%;">PyTorch/Train</span></td>
<td id="S5.T2.fig1.3.3.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.3.2.1" class="ltx_text" style="font-size:90%;">R</span></td>
<td id="S5.T2.fig1.3.3.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.3.3.1" class="ltx_text" style="font-size:90%;">84.94</span></td>
<td id="S5.T2.fig1.3.3.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.3.4.1" class="ltx_text" style="font-size:90%;">13.21</span></td>
<td id="S5.T2.fig1.3.3.5" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.3.5.1" class="ltx_text" style="font-size:90%;">  1.33</span></td>
<td id="S5.T2.fig1.3.3.6" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.3.6.1" class="ltx_text" style="font-size:90%;">  0.52</span></td>
</tr>
<tr id="S5.T2.fig1.3.4" class="even ltx_tr">
<td rowspan="4" id="S5.T2.fig1.3.4.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.4.1.1" class="ltx_text" style="font-size:90%;">PyTorch/Infer</span></td>
<td id="S5.T2.fig1.3.4.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.4.2.1" class="ltx_text" style="font-size:90%;">R</span></td>
<td id="S5.T2.fig1.3.4.3" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.4.3.1" class="ltx_text" style="font-size:90%;">83.96</span></td>
<td id="S5.T2.fig1.3.4.4" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.4.4.1" class="ltx_text" style="font-size:90%;">14.01</span></td>
<td id="S5.T2.fig1.3.4.5" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.4.5.1" class="ltx_text" style="font-size:90%;">  1.86</span></td>
<td id="S5.T2.fig1.3.4.6" class="ltx_td ltx_align_center ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.4.6.1" class="ltx_text" style="font-size:90%;">  0.17</span></td>
</tr>
<tr id="S5.T2.fig1.3.5" class="odd ltx_tr">
<td id="S5.T2.fig1.3.5.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.5.1.1" class="ltx_text" style="font-size:90%;">V</span></td>
<td id="S5.T2.fig1.3.5.2" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.5.2.1" class="ltx_text" style="font-size:90%;">83.56</span></td>
<td id="S5.T2.fig1.3.5.3" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.5.3.1" class="ltx_text" style="font-size:90%;">  6.69</span></td>
<td id="S5.T2.fig1.3.5.4" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.5.4.1" class="ltx_text" style="font-size:90%;">  9.75</span></td>
<td id="S5.T2.fig1.3.5.5" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.5.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T2.fig1.3.6" class="even ltx_tr">
<td id="S5.T2.fig1.3.6.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.6.1.1" class="ltx_text" style="font-size:90%;">I</span></td>
<td id="S5.T2.fig1.3.6.2" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.6.2.1" class="ltx_text" style="font-size:90%;">69.50</span></td>
<td id="S5.T2.fig1.3.6.3" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.6.3.1" class="ltx_text" style="font-size:90%;">20.45</span></td>
<td id="S5.T2.fig1.3.6.4" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.6.4.1" class="ltx_text" style="font-size:90%;">  9.82</span></td>
<td id="S5.T2.fig1.3.6.5" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.6.5.1" class="ltx_text" style="font-size:90%;">  0.23</span></td>
</tr>
<tr id="S5.T2.fig1.3.7" class="odd ltx_tr">
<td id="S5.T2.fig1.3.7.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.7.1.1" class="ltx_text" style="font-size:90%;">D</span></td>
<td id="S5.T2.fig1.3.7.2" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.7.2.1" class="ltx_text" style="font-size:90%;">60.81</span></td>
<td id="S5.T2.fig1.3.7.3" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.7.3.1" class="ltx_text" style="font-size:90%;">34.24</span></td>
<td id="S5.T2.fig1.3.7.4" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.7.4.1" class="ltx_text" style="font-size:90%;">  4.94</span></td>
<td id="S5.T2.fig1.3.7.5" class="ltx_td ltx_align_center" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.7.5.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
<tr id="S5.T2.fig1.3.8" class="even ltx_tr">
<td id="S5.T2.fig1.3.8.1" class="ltx_td ltx_align_left ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.8.1.1" class="ltx_text" style="font-size:90%;">llama.cpp</span></td>
<td id="S5.T2.fig1.3.8.2" class="ltx_td ltx_align_left ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.8.2.1" class="ltx_text" style="font-size:90%;">L</span></td>
<td id="S5.T2.fig1.3.8.3" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.8.3.1" class="ltx_text" style="font-size:90%;">59.84</span></td>
<td id="S5.T2.fig1.3.8.4" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.8.4.1" class="ltx_text" style="font-size:90%;">38.51</span></td>
<td id="S5.T2.fig1.3.8.5" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.8.5.1" class="ltx_text" style="font-size:90%;">  1.65</span></td>
<td id="S5.T2.fig1.3.8.6" class="ltx_td ltx_align_center ltx_border_bb ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S5.T2.fig1.3.8.6.1" class="ltx_text" style="font-size:90%;">  0.00</span></td>
</tr>
</tbody>
</table>
<br />

<figcaption><span class="ltx_tag ltx_tag_table">Table 2: </span><span id="S5.T2.fig1.2.1" class="ltx_text ltx_font_italic" style="font-size:90%;">Typical memory access types in different GPU workloads.</span></figcaption>
</figure>

<div id="S5.SS2.p5" class="ltx_para">

A single kernel may exhibit a combination of these three patterns. Together, our three templates cover nearly all memory access patterns found in typical GPU workloads. The remaining cases (less than 1%) arise from indirect memory access (pointer-chasing), where the base address originates from a value in GPU memory rather than from arguments—a behavior that is extremely rare in GPU workloads \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">phos-sosp25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">picker-arxiv24</span>\].

</div>

<div id="S5.SS2.p6" class="ltx_para ltx_noindent">

<span id="S5.SS2.p6.1.1" class="ltx_text ltx_font_bold">Memory analyzer.</span> Motivated by these findings, we build an offline memory analyzer that consumes the profiled per-kernel memory traces and launch arguments, and derives the mapping between them according to the three templates. The analyzer first scans the kernel’s argument list to identify 64-bit integer values that appear as beginning addresses of the memory regions in the trace. Next, it attempts to match the <span id="S5.SS2.p6.1.2" class="ltx_text ltx_font_italic">spatial extent</span> of these regions against the three templates in order. If the region size is invariant across all invocations of this kernel, it is classified as T1. Otherwise, the analyzer enumerates the remaining 64- and 32-bit integer arguments, as well as their combinations (products), to check for linear proportionality with the region’s size (T2) or stride (T3). Upon a successful match, the analyzer records the specific argument indices and the corresponding linear coefficients to formulate the prediction rules. Note that for C-style struct arguments, the analyzer slices them into 64- and 32-bit integers and treats each of them as an independent candidate. Since the total number of arguments is small (ranging from a few to dozens), the analysis can typically complete within seconds.

</div>

<div id="S5.SS2.p7" class="ltx_para ltx_noindent">

<span id="S5.SS2.p7.1.1" class="ltx_text ltx_font_bold">Memory access prediction.</span> We implemented an online predictor as a preloaded DLL. It intercepts all GPU command launch APIs and utilizes the captured runtime launch arguments to evaluate the offline-derived prediction formulas, thereby calculating the precise memory access regions of each command (within a microsecond). The predictor then aligns these accessed regions to page boundaries, attaches the prediction results to the command metadata, and forwards them to the memory manager for proactive scheduling.

</div>

<div id="S5.SS2.p8" class="ltx_para ltx_noindent">

<span id="S5.SS2.p8.1.1" class="ltx_text ltx_font_bold">Prediction accuracy.</span> We evaluate the accuracy across scientific and ML workloads. While the naive allocation-based approach captures all accesses, it incurs a high false positive rate due to coarse-grained prediction. In contrast, our template-based prediction achieves near-perfect coverage (0.25% average false negative) with zero false positives. This ensures that <span id="S5.SS2.p8.1.2" class="ltx_text">MSched</span> precisely identifies the actual working set without wasting resources on dormant data. Note that <span id="S5.SS2.p8.1.3" class="ltx_text">MSched</span> retains demand paging as a fallback. The rare false negatives are handled transparently via standard page faults, ensuring execution correctness with negligible performance overhead.

</div>

</div>

</div>

<div id="S6" class="section ltx_section">

## Proactive Memory Scheduling

<div id="S6.SS1" class="section ltx_subsection">

### Task Scheduler

<div id="S6.SS1.p1" class="ltx_para">

To implement the extended context switch and proactive memory scheduling, <span id="S6.SS1.p1.1.1" class="ltx_text">MSched</span> extends XSched \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\], an open-source GPU scheduler. XSched leverages the GPU driver’s timeslice group (TSG) \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-tsg</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gcaps-ecrts24</span>\] control interface to preemptively suspend the running task and perform context switching to the next one based on a user-defined scheduling policy. <span id="S6.SS1.p1.1.2" class="ltx_text">MSched</span> augments the context switch routine: upon suspending the current task and saving its architectural state, the scheduler invokes the <span id="S6.SS1.p1.1.3" class="ltx_text">MSched</span> memory manager. The manager then evicts inactive pages to create sufficient space and populates the predicted working set of the next task into HBM.

</div>

<div id="S6.SS1.p2" class="ltx_para">

However, achieving efficient memory scheduling requires not only spatially precise working set prediction, but also <span id="S6.SS1.p2.1.1" class="ltx_text ltx_framed ltx_framed_underline">temporal accuracy</span>, as the accessing order directly determines the optimal page placement. According to Belady’s algorithm \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">opt-1966</span>\], the theoretically optimal strategy (OPT) is to evict the page that will not be referenced for the longest time. The asynchronous and sequential nature of GPU tasks, combined with our template-based prediction technique, offers a unique opportunity to implement this policy—an objective that is elusive in the CPU world due to the opaque execution flows.

</div>

<div id="S6.SS1.p3" class="ltx_para">

Exploiting this opportunity is non-trivial under multitasking. While the execution latency of individual GPU commands (kernels and memory copies) is deterministic and stable \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">rammer-osdi20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">clockwork-osdi20</span>\], context switching between tasks fundamentally alters the global execution timeline. The interleaving of commands from concurrent tasks disrupts the sequential continuity observed in individual tasks. Consequently, without awareness of the scheduling plan, it is impossible to accurately predict the absolute timing of future memory accesses, thereby undermining the implementation of the OPT policy.

</div>

<div id="S6.SS1.p4" class="ltx_para ltx_noindent">

<span id="S6.SS1.p4.1.1" class="ltx_text ltx_font_bold">Task scheduling timeline is the Rosetta Stone.</span> To address this challenge, <span id="S6.SS1.p4.1.2" class="ltx_text">MSched</span> co-designs the task scheduler and memory manager. We modified the scheduling policy module to expose its task scheduling timeline as an additional argument when the context switcher invokes the memory manager. The task scheduling timeline is an ordered sequence of task entries and allocated timeslices akin to the run queue in OS schedulers \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">modern-os</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">linux-runqueue</span>\]. In <span id="S6.SS1.p4.1.3" class="ltx_text">MSched</span>, this simple structure plays a pivotal role. It provides the ground truth for the future execution timeline—which task will execute, for how long, and in what order. This enables the memory manager to deterministically resolve the global memory access sequence and enforce the optimal replacement policy, reducing overall migration volume. Moreover, the timeline is easy to generate and effectively decouples the scheduling policy from memory management, allowing <span id="S6.SS1.p4.1.4" class="ltx_text">MSched</span> to support diverse policies.

</div>

</div>

<div id="S6.SS2" class="section ltx_subsection">

### Memory Manager

<div id="S6.SS2.p1" class="ltx_para">

Armed with the requisite inputs—accurate working set from the predictor, command execution latency from the kernel profiler, and the task timeline from the scheduler—<span id="S6.SS2.p1.1.1" class="ltx_text">MSched</span> is able to implement efficient proactive memory scheduling. Fig. <a href="https://arxiv.org/html/2512.24637v2#S6.F4.fig1" class="ltx_ref" title="Figure 4 ‣ Memory Manager ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">4</span></a> illustrates the memory scheduling logic of <span id="S6.SS2.p1.1.2" class="ltx_text">MSched</span>.

</div>

<div id="S6.SS2.p2" class="ltx_para ltx_noindent">

<span id="S6.SS2.p2.1.1" class="ltx_text ltx_font_bold">Distributed metadata management.</span> To minimize runtime overhead, <span id="S6.SS2.p2.1.2" class="ltx_text">MSched</span> adopts a distributed design. The memory manager is composed of a centralized <span id="S6.SS2.p2.1.3" class="ltx_text ltx_framed ltx_framed_underline">coordinator</span> daemon and a per-process <span id="S6.SS2.p2.1.4" class="ltx_text ltx_framed ltx_framed_underline">helper</span> library. The helper, loaded into each application process, records each intercepted GPU command and its predicted working set (pages) into a process-local command queue. It also attaches the offline-profiled execution latency to each command. This allows each helper to maintain a precise, local sequence of future memory accesses relative to its own execution flow, without flooding the central coordinator with fine-grained metadata.

</div>

<div id="S6.SS2.p3" class="ltx_para ltx_noindent">

<span id="S6.SS2.p3.1.1" class="ltx_text ltx_font_bold">Driver support.</span> <span id="S6.SS2.p3.1.2" class="ltx_text">MSched</span> augments the <span id="S6.SS2.p3.1.3" class="ltx_text ltx_font_typewriter">ioctl</span> interfaces of the GPU kernel-mode driver (KMD) to manipulate its internal LRU page eviction list. Under page faults with full HBM, the driver will evict pages from the list head. The new <span id="S6.SS2.p3.1.4" class="ltx_text ltx_font_typewriter">madvise</span> interface allows userspace to move specific pages to the tail of the list, protecting them from immediate eviction. <span id="S6.SS2.p3.1.5" class="ltx_text">MSched</span> also adds a migrate engine to the KMD. Upon a <span id="S6.SS2.p3.1.6" class="ltx_text ltx_font_typewriter">migrate</span> call, the migrate engine evicts head pages of the eviction list to host CPU DRAM, freeing up enough space, then proactively populates the specified memory pages into GPU HBM.

</div>

<figure id="S6.F4.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div id="S6.F4.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<img src="x4.png" id="S6.F4.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="598" height="145" alt="Refer to caption" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 4: </span><em>The memory manager of <span id="S6.F4.fig1.3.1.1" class="ltx_text ltx_font_upright">MSched</span>.</em></figcaption>
</figure>

<div id="S6.SS2.p4" class="ltx_para ltx_noindent">

<span id="S6.SS2.p4.1.1" class="ltx_text ltx_font_bold">Enforcing the OPT memory scheduling.</span> The coordinator resolves inter-task interleaving using the scheduler’s task timeline. Upon a context switch, the scheduler informs the coordinator of the current timeline, which includes ordered task IDs, currently executing commands, and allocated timeslices of each task. According to Belady’s OPT algorithm \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">opt-1966</span>\], the ideal eviction order is the reverse of the access order. Therefore, the coordinator iterates through the timeline in <span id="S6.SS2.p4.1.2" class="ltx_text ltx_font_italic">reverse order</span> and, via shared-memory IPC, instructs each helper to <span id="S6.SS2.p4.1.3" class="ltx_text ltx_font_typewriter">madvise</span> the pages accessed within its assigned timeslice. In the case of Fig. <a href="https://arxiv.org/html/2512.24637v2#S6.F4.fig1" class="ltx_ref" title="Figure 4 ‣ Memory Manager ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">4</span></a>, the pages in the eviction list would eventually become, in order: unreferenced across the timeline (grey), Task3’s working set within 30 ms (orange), Task2’s working set in 10 ms (pink), and Task1’s working set in 20 ms (cyan). Consequently, the list head naturally exposes the optimal eviction candidates (pages not needed for the longest time), while the near-term working set resides safely at the tail. Once ordered, the coordinator instructs the helper of the next-to-run task to call <span id="S6.SS2.p4.1.4" class="ltx_text ltx_font_typewriter">migrate</span> to populate its immediate working set (cyan), while evicting idle pages (grey), completing a context switch augmented with memory scheduling.

</div>

<div id="S6.SS2.p5" class="ltx_para">

The optimal eviction order (i.e., page scheduling list in Fig. <a href="https://arxiv.org/html/2512.24637v2#S6.F4.fig1" class="ltx_ref" title="Figure 4 ‣ Memory Manager ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">4</span></a>) is dynamic. First, the task timeline may reorder due to the policy’s decisions. Second, workloads like model training, iterative LLM decoding, and Mixture-of-Experts (MoE) models intermittently launch new GPU commands, causing the predicted memory page sets to evolve dynamically. Therefore, the memory manager must perform the complete procedure at every context switch to timely apply the latest page scheduling list to the driver’s eviction list. In practice, this frequency keeps the eviction order effectively optimal, as evidenced by the evaluation results with LLM decoding (Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F7.fig1" class="ltx_ref" title="Figure 7 ‣ End-to-end Application Performance ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7</span></a> (d)) and DNN training (Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F13.fig1" class="ltx_ref" title="Figure 13 ‣ Comparison with Existing Systems ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">13</span></a> (b)).

</div>

</div>

<div id="S6.SS3" class="section ltx_subsection">

### Page Migration Pipeline

<figure id="S6.F5.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div id="S6.F5.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<embed src="index.html" id="S6.F5.1.g1" class="ltx_graphics ltx_centering ltx_missing ltx_missing_image" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 5: </span><em>Page migration pipeline and early start optimizations.</em></figcaption>
</figure>

<div id="S6.SS3.p1" class="ltx_para">

Since page eviction and population dominate the context switching latency, maximizing the migration throughput is critical. We observe that modern GPU architectures provide substantial hardware parallelism, featuring multiple Copy Engines (CEs) capable of concurrent DMA operations \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">cuda-guide</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">amd-ce</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">demystify-rtas24</span>\]. Furthermore, interconnects today like PCIe and NVLink-C2C support full-duplex communication \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">GH200-study-arxiv24</span>\], theoretically enabling parallel Device-to-Host eviction (D2H) and Host-to-Device population (H2D). However, we find that the standard migration mechanisms fail to exploit these capabilities, leaving significant bandwidth potential untapped.

</div>

<figure id="S6.F6.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_3">
<div id="S6.F6.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:138.8pt;">
<img src="x6.png" id="S6.F6.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="457" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_cell ltx_flex_size_3">
<div id="S6.F6.2" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:138.8pt;">
<img src="x7.png" id="S6.F6.2.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="457" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_cell ltx_flex_size_3">
<div id="S6.F6.3" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:138.8pt;">
<img src="x8.png" id="S6.F6.3.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="457" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
</div>
<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 6: </span><em>Comparison of (a) end-to-end throughput, (b) page fault count per task completion, and (c) memory migration volume per task completion, between native demand paging (UM), proactive memory scheduling (<span id="S6.F6.fig1.5.1.1" class="ltx_text ltx_font_upright">MSched</span>), and theoretical optimal limit (Ideal).</em></figcaption>
</figure>

<div id="S6.SS3.p2" class="ltx_para">

Fig. <a href="https://arxiv.org/html/2512.24637v2#S6.F5.fig1" class="ltx_ref" title="Figure 5 ‣ Page Migration Pipeline ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">5</span></a> illustrates the baseline migration workflow in the latest GPU drivers \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-driver-580</span>\]. Swapping a page typically entails a serialized sequence of four operations: unmapping the victim page from the GPU page table using a specific command, evicting the victim page to the host (D2H) to reclaim space, populating the target page to the device (H2D), and finally establishing the new mapping. Consequently, the execution of subsequent compute kernels is strictly stalled until the migration of the entire working set completes.

</div>

<div id="S6.SS3.p3" class="ltx_para">

To maximize memory scheduling throughput, <span id="S6.SS3.p3.1.1" class="ltx_text">MSched</span> implements an efficient page migration pipeline that exploits parallelism between CEs. Concretely, the migrate engine initiates the unmapping and D2H eviction of the first victim page on CE0. Once space is reclaimed, it immediately launches H2D population and mapping of the first target page on CE1 while CE0 proceeds with unmapping and D2H for the second page. This forms a tight pipeline that performs eviction and population in parallel, effectively saturating the full-duplex interconnect bandwidth.

</div>

<div id="S6.SS3.p4" class="ltx_para">

Furthermore, <span id="S6.SS3.p4.1.1" class="ltx_text">MSched</span> orders page migrations in the predicted access order. This enables <span id="S6.SS3.p4.1.2" class="ltx_text ltx_font_italic">early execution</span>: rather than stalling for the entire working set migration, the GPU compute kernel begins execution as soon as its immediate dependency pages are resident. To orchestrate the pipeline dependencies without incurring CPU intervention overhead, <span id="S6.SS3.p4.1.3" class="ltx_text">MSched</span> utilizes hardware-supported synchronization primitives—GPU trackers and events (two kinds of semaphores on GPUs)—to enforce fine-grained signaling between CEs and CUs. This pipelined design improves hardware utilization and reduces the overhead of proactive memory scheduling.

</div>

</div>

</div>

<div id="S7" class="section ltx_section">

## Evaluation

<div id="S7.p1" class="ltx_para ltx_noindent">

<span id="S7.p1.1.1" class="ltx_text ltx_font_bold">Experimental setup.</span> The evaluation is mainly conducted on a server equipped with Intel Core Ultra 9 285K (24 cores), 96 GB of DDR5 memory, and an NVIDIA RTX 5080 GPU (16 GB of HBM and PCIe 5.0$\times$<!-- -->16). The system runs Ubuntu 24.04 with NVIDIA driver 580.95.05 and CUDA toolkit 12.9 installed. <span id="S7.p1.1.2" class="ltx_text">MSched</span> extends XSched \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\], an open-source GPU scheduler, and adopts its round-robin (RR) scheduling policy to multiplex all tasks in the system, matching the default time-sharing behavior of commodity GPUs. Note that <span id="S7.p1.1.3" class="ltx_text">MSched</span> operates at OS-level and is fully transparent to applications, enabling proactive memory scheduling without any source code modifications or recompilation.

</div>

<div id="S7.SS1" class="section ltx_subsection">

### Effectiveness of Proactive Memory Scheduling

<div id="S7.SS1.p1" class="ltx_para ltx_noindent">

<span id="S7.SS1.p1.1.1" class="ltx_text ltx_font_bold">Methodology.</span> We first evaluate the mechanistic advantages of proactive memory scheduling over demand paging. We handcraft two kinds of representative GPU tasks: a vector addition task that touches large memory regions within short bursts, and a matrix multiplication task which is compute-bound with high arithmetic intensity. We launch two processes for each type (four concurrent processes in total), continuously issuing tasks. By adjusting vector lengths and number of matrices to be computed, we precisely control the aggregate memory footprint and ensure equal memory consumption across all tasks. We compare <span id="S7.SS1.p1.1.2" class="ltx_text">MSched</span>’s proactive scheduling against the native demand paging (CUDA UM) and an <span id="S7.SS1.p1.1.3" class="ltx_text ltx_font_italic">Ideal</span> baseline, which represents the theoretical performance upper bound calculated using the strict OPT page replacement policy and the hardware’s actual performance metrics.

</div>

<div id="S7.SS1.p2" class="ltx_para ltx_noindent">

<span id="S7.SS1.p2.1.1" class="ltx_text ltx_font_bold">Performance breakdown.</span> Fig. <a href="https://arxiv.org/html/2512.24637v2#S6.F6.fig1" class="ltx_ref" title="Figure 6 ‣ Page Migration Pipeline ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">6</span></a> reports the end-to-end task throughput (normalized to exclusive in-HBM execution without oversubscription), the average number of page faults per task completion, and the average memory migration volume per task completion. As observed, demand paging exhibits a performance cliff immediately upon memory oversubscription (allocated memory over 100%), suffering a staggering 16.2$\times$ throughput collapse. This stems from typical GPU workloads’ poor locality and large working sets whose aggregate exceeds HBM capacity, inducing tens of thousands of page faults, severe thrashing, and a surge in total migration. Notably, the recorded migration volume for demand paging is not strictly equal to the product of fault counts and page size (4KB). This is because CUDA UM attempts to prefetch data without precise knowledge of the working set.

</div>

<div id="S7.SS1.p3" class="ltx_para">

In contrast, proactive memory scheduling delivers substantial benefits. At 200% memory subscription, <span id="S7.SS1.p3.1.1" class="ltx_text">MSched</span> achieves a 9.67$\times$ speedup over demand paging. Even under extreme pressure (300% usage), it retains 43.4% of the original in-HBM throughput. This performance gain is attributed to the elimination of page faults (only sporadic occurrences) and the enforcement of the global OPT placement policy, which minimizes the total data movement. Moreover, <span id="S7.SS1.p3.1.2" class="ltx_text">MSched</span>’s performance is close to the theoretical limit (<span id="S7.SS1.p3.1.3" class="ltx_text ltx_font_italic">Ideal</span>), indicating efficient utilization of the hardware. Notably, without oversubscription (100%), <span id="S7.SS1.p3.1.4" class="ltx_text">MSched</span> retains 99.41% throughput, confirming its negligible runtime overhead (0.59%).

</div>

</div>

<div id="S7.SS2" class="section ltx_subsection">

### End-to-end Application Performance

<figure id="S7.T3.fig1" class="ltx_table ltx_figure_panel ltx_minipage ltx_align_center ltx_align_middle" style="width:433.6pt;">
<br />

<table id="S7.T3.fig1.3" class="ltx_tabular ltx_centering ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<tbody>
<tr id="S7.T3.fig1.3.1" class="odd ltx_tr">
<td id="S7.T3.fig1.3.1.1" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.1.1.1" class="ltx_text" style="font-size:90%;">Comb.</span></td>
<td id="S7.T3.fig1.3.1.2" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.1.2.1" class="ltx_text" style="font-size:90%;">Type</span></td>
<td id="S7.T3.fig1.3.1.3" class="ltx_td ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"></td>
<td id="S7.T3.fig1.3.1.4" class="ltx_td ltx_align_left ltx_border_tt" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.1.4.1" class="ltx_text" style="font-size:90%;">Task</span></td>
</tr>
<tr id="S7.T3.fig1.3.2" class="even ltx_tr">
<td id="S7.T3.fig1.3.2.1" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.2.1.1" class="ltx_text" style="font-size:90%;">A</span></td>
<td id="S7.T3.fig1.3.2.2" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.2.2.1" class="ltx_text" style="font-size:90%;">SciComp</span></td>
<td id="S7.T3.fig1.3.2.3" class="ltx_td ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"></td>
<td id="S7.T3.fig1.3.2.4" class="ltx_td ltx_align_left ltx_border_t" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.2.4.1" class="ltx_text" style="font-size:90%;">dwt2d, hotspot, cfd, nn</span></td>
</tr>
<tr id="S7.T3.fig1.3.3" class="odd ltx_tr">
<td id="S7.T3.fig1.3.3.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.3.1.1" class="ltx_text" style="font-size:90%;">B</span></td>
<td id="S7.T3.fig1.3.3.2" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.3.2.1" class="ltx_text" style="font-size:90%;">MultiDNN</span></td>
<td id="S7.T3.fig1.3.3.3" class="ltx_td" style="padding-top: 0.25pt; padding-bottom: 0.25pt"></td>
<td id="S7.T3.fig1.3.3.4" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.3.4.1" class="ltx_text" style="font-size:90%;">RNet, VGG, Inception, DNet</span></td>
</tr>
<tr id="S7.T3.fig1.3.4" class="even ltx_tr">
<td id="S7.T3.fig1.3.4.1" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.4.1.1" class="ltx_text" style="font-size:90%;">C</span></td>
<td id="S7.T3.fig1.3.4.2" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.4.2.1" class="ltx_text" style="font-size:90%;">HybridDL</span></td>
<td id="S7.T3.fig1.3.4.3" class="ltx_td" style="padding-top: 0.25pt; padding-bottom: 0.25pt"></td>
<td id="S7.T3.fig1.3.4.4" class="ltx_td ltx_align_left" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.4.4.1" class="ltx_text" style="font-size:90%;">RNet, VGG, Inception, DNet, Llama3</span></td>
</tr>
<tr id="S7.T3.fig1.3.5" class="odd ltx_tr">
<td id="S7.T3.fig1.3.5.1" class="ltx_td ltx_align_left ltx_border_bb" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.5.1.1" class="ltx_text" style="font-size:90%;">D</span></td>
<td id="S7.T3.fig1.3.5.2" class="ltx_td ltx_align_left ltx_border_bb" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.5.2.1" class="ltx_text" style="font-size:90%;">MultiLLM</span></td>
<td id="S7.T3.fig1.3.5.3" class="ltx_td ltx_border_bb" style="padding-top: 0.25pt; padding-bottom: 0.25pt"></td>
<td id="S7.T3.fig1.3.5.4" class="ltx_td ltx_align_left ltx_border_bb" style="padding-top: 0.25pt; padding-bottom: 0.25pt"><span id="S7.T3.fig1.3.5.4.1" class="ltx_text" style="font-size:90%;">Llama3 (multiple instances)</span></td>
</tr>
</tbody>
</table>
<br />

<figcaption><span class="ltx_tag ltx_tag_table">Table 3: </span><span id="S7.T3.fig1.2.1" class="ltx_text ltx_font_italic" style="font-size:90%;">GPU task combinations tested in §<a href="https://arxiv.org/html/2512.24637v2#S7.SS2" class="ltx_ref" title="End-to-end Application Performance ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7.2</span></a>.</span></figcaption>
</figure>

<div id="S7.SS2.p1" class="ltx_para ltx_noindent">

<span id="S7.SS2.p1.1.1" class="ltx_text ltx_font_bold">Workloads.</span> We evaluate <span id="S7.SS2.p1.1.2" class="ltx_text">MSched</span> against native demand paging (UM) under real applications using four task combinations as listed in Table <a href="https://arxiv.org/html/2512.24637v2#S7.T3.fig1" class="ltx_ref" title="Table 3 ‣ End-to-end Application Performance ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">3</span></a>. <span id="S7.SS2.p1.1.3" class="ltx_text ltx_font_bold">A (SciComp)</span>: four representative scientific computing tasks from Rodinia \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">rodinia-iiswc09</span>\] benchmark suite—2D discrete wavelet transform (dwt2d), thermal simulation (hotspot), fluid dynamics solver (cfd), and nearest neighbors search (nn). <span id="S7.SS2.p1.1.4" class="ltx_text ltx_font_bold">B (MultiDNN)</span>: inference tasks of four classic DNNs using PyTorch—ResNet152(RNet), VGG19, InceptionV3, and DenseNet201 (DNet). <span id="S7.SS2.p1.1.5" class="ltx_text ltx_font_bold">C (HybridDL)</span>: all tasks in B, combined with an LLM inference task (int8-quantized Llama3-8B via llama.cpp). <span id="S7.SS2.p1.1.6" class="ltx_text ltx_font_bold">D (MultiLLM)</span>: Multiple concurrent Llama3-8B inference instances. Combinations A, B, and C are relatively compute-intensive, whereas D is memory-bound and sweeps large memory regions within short time windows. Each kind of task runs as an independent process. For each combination, we measure the end-to-end performance under three memory oversubscription pressures: <span id="S7.SS2.p1.1.7" class="ltx_text ltx_font_bold">Light</span> (total memory allocation = 150% of HBM capacity), <span id="S7.SS2.p1.1.8" class="ltx_text ltx_font_bold">Medium</span> (200%), and <span id="S7.SS2.p1.1.9" class="ltx_text ltx_font_bold">Heavy</span> (300%). We control the total memory footprints by scaling problem sizes (for combination A), adjusting inference batch sizes (for B and C), and increasing the concurrent model instance count (for D). Across all testcases, we balance memory usage across tasks as evenly as possible. We then measure task completion throughput for A and DNN models, and decoding throughput for LLMs.

</div>

<figure id="S7.F7.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F7.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:212.5pt;">
<img src="x9.png" id="S7.F7.1.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="726" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F7.2" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:212.5pt;">
<img src="x10.png" id="S7.F7.2.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="726" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F7.3" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:212.5pt;">
<img src="x11.png" id="S7.F7.3.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="726" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F7.4" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:212.5pt;">
<img src="x12.png" id="S7.F7.4.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="726" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
</div>
<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 7: </span><em>Comparison of end-to-end throughput between UM and <span id="S7.F7.fig1.6.1.1" class="ltx_text ltx_font_upright">MSched</span> under different workload setups and memory pressures.</em></figcaption>
</figure>

<div id="S7.SS2.p2" class="ltx_para ltx_noindent">

<span id="S7.SS2.p2.1.1" class="ltx_text ltx_font_bold">Performance.</span> Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F7.fig1" class="ltx_ref" title="Figure 7 ‣ End-to-end Application Performance ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7</span></a> depicts the end-to-end throughput across all workload sets, normalized to in-HBM execution. Under memory oversubscription, the native demand paging suffers a catastrophic performance collapse. For the compute-heavy A, B, and C, throughput plummets to an average of 6.19% of in-HBM execution. The degradation is even more severe for the memory-intensive LLM workloads (D), where throughput drops to a mere 1.29%, rendering the GPU practically unusable due to severe page thrashing and IO stalls.

</div>

<div id="S7.SS2.p3" class="ltx_para">

In contrast, <span id="S7.SS2.p3.6.1" class="ltx_text">MSched</span> consistently delivers superior performance across all workloads. For combinations A, B, and C, <span id="S7.SS2.p3.6.2" class="ltx_text">MSched</span> achieves average speedups of 11.05$\times$, 9.35$\times$, and 7.52$\times$ under Light, Medium, and Heavy memory pressures, respectively, compared to demand paging. For the LLM workloads (D), the gains are even more pronounced, reaching 57.88$\times$, 44.79$\times$, and 33.60$\times$ improvements. These results align with the micro-benchmark findings in Fig. <a href="https://arxiv.org/html/2512.24637v2#S6.F6.fig1" class="ltx_ref" title="Figure 6 ‣ Page Migration Pipeline ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">6</span></a> (a) and closely approach the theoretical optimal limit, confirming the fundamental advantages of proactive memory scheduling in supporting concurrent GPU tasks under memory pressure.

</div>

</div>

<div id="S7.SS3" class="section ltx_subsection">

### Ablation Study

<div id="S7.SS3.p1" class="ltx_para ltx_noindent">

<span id="S7.SS3.p1.1.1" class="ltx_text ltx_font_bold">Impact of prediction accuracy.</span> Table <a href="https://arxiv.org/html/2512.24637v2#S5.T1.fig1" class="ltx_ref" title="Table 1 ‣ Naive Solution: Allocation-granularity Prediction ‣ Memory Access Prediction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">1</span></a> in §<a href="https://arxiv.org/html/2512.24637v2#S5.T2.fig1" class="ltx_ref" title="Table 2 ‣ Our Approach: Template-based Prediction ‣ Memory Access Prediction ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">2</span></a> evaluates the prediction accuracy of the naive allocation-granularity prediction and our template-based prediction under different GPU workloads. While the naive solution maintains a low false negative rate (covers almost all accesses), it suffers from a high false positive rate (most predicted memory are not actually accessed). In contrast, our approach attains near-perfect coverage with zero false positives due to strict template matching. Here, we quantify how this precision gap translates into migration efficiency and application performance. We execute the LLM inference workloads (Combination D in §<a href="https://arxiv.org/html/2512.24637v2#S7.SS2" class="ltx_ref" title="End-to-end Application Performance ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7.2</span></a>) using both prediction strategies and measure the average memory migration volume per decode step and the overall throughput.

</div>

<figure id="S7.F8.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F8.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:208.1pt;">
<img src="x13.png" id="S7.F8.1.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="726" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F8.2" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:208.1pt;">
<img src="x14.png" id="S7.F8.2.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="726" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
</div>
<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 8: </span><em>Comparison of (a) memory migration volume and (b) end-to-end throughput between allocation-granularity prediction and template-based prediction under different memory pressures.</em></figcaption>
</figure>

<div id="S7.SS3.p2" class="ltx_para">

As shown in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F8.fig1" class="ltx_ref" title="Figure 8 ‣ Ablation Study ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">8</span></a>, under Light and Medium memory pressures, the naive allocation-granularity prediction incurs a 4.77$\times$ inflation in migration volume relative to our template-based method, resulting in 5.17$\times$ and 5.39$\times$ degradation in end-to-end throughput, respectively. This penalty arises primarily from excessive, erroneous preloading that wastes interconnect bandwidth. Furthermore, under Heavy pressure, this inefficiency compounds: migration inflation surges to 12.27$\times$. In this case, over-prediction pollutes the scarce HBM with useless data and displaces active working sets of subsequent tasks, precipitating a cascade of eviction and repopulation thrashing. This effect is further amplified under high memory pressure. Consequently, the naive approach suffers a 15.67$\times$ throughput drop, nearly erasing the benefits of proactive scheduling. These results underscore that precise working set prediction is indispensable—<span id="S7.SS3.p2.5.1" class="ltx_text ltx_font_italic">accuracy</span> directly governs migration efficiency and ultimately determines system-level memory scheduling performance under oversubscription.

</div>

<figure id="S7.F9.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F9.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:260.2pt;">
<img src="x15.png" id="S7.F9.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="555" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F9.2" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:169.1pt;">
<img src="x16.png" id="S7.F9.2.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="853" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
</div>
<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 9: </span><em>(a) Page migration bandwidth using different methods on two different platforms, and (b) comparison of end-to-end throughput between with (w/) and without (w/o) pipelined migration.</em></figcaption>
</figure>

<div id="S7.SS3.p3" class="ltx_para ltx_noindent">

<span id="S7.SS3.p3.1.1" class="ltx_text ltx_font_bold">Effectiveness of pipelined migration.</span> We further analyze the impact of the page migration bandwidth between GPU HBM and host CPU DRAM on memory scheduling and evaluate the efficacy of our pipelined migration mechanism. To demonstrate the generality of our approach across different platforms, we add a <span id="S7.SS3.p3.1.2" class="ltx_text ltx_font_bold">new testbed</span> equipped with Intel Core i7-13700, 64 GB DDR4 memory, and an NVIDIA <span id="S7.SS3.p3.1.3" class="ltx_text ltx_font_bold">RTX 3080</span> GPU (10 GB HBM, PCIe 4.0$\times$<!-- -->16).

</div>

<div id="S7.SS3.p4" class="ltx_para">

We first measure the bandwidth of page eviction (D2H including unmapping), page population (H2D including mapping), page swapping (evict then populate) through page faults, and proactive page swapping with and without our pipelined migration technique. As shown in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F9.fig1" class="ltx_ref" title="Figure 9 ‣ Ablation Study ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">9</span></a> (a), the effective swap bandwidth without pipeline is limited to the average of D2H and H2D, which is 41.7 GB/s on RTX 5080 and 22.22 GB/s on RTX 3080. In contrast, our pipelined approach overlaps D2H and H2D to exploit the full-duplex interconnect, boosting the effective swap bandwidth to 63.5 GB/s (1.52$\times$ speedup) on RTX 5080 and 39.8 GB/s (1.79$\times$ speedup) on RTX 3080. Note that throughput on RTX 5080 is still far below the theoretical ceiling of the PCIe 5.0$\times$<!-- -->16 link (64 GB/s$\times$<!-- -->2). We identify a hardware bottleneck between the PCIe root complex and the DRAM in the host CPU. This is a known limitation in the chiplet design of recent Intel desktop CPUs \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">arrowlake-bottleneck</span>\], where the network on-chip (NoC) throttles the traffic between the IO die and the DDR5 controller. This issue is absent in other CPU families or server-grade platforms.

</div>

<div id="S7.SS3.p5" class="ltx_para">

Next, we assess the end-to-end impact using the same workload from §<a href="https://arxiv.org/html/2512.24637v2#S7.SS1" class="ltx_ref" title="Effectiveness of Proactive Memory Scheduling ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7.1</span></a> on RTX 5080. As illustrated in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F9.fig1" class="ltx_ref" title="Figure 9 ‣ Ablation Study ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">9</span></a> (b), the improved bandwidth translates directly to performance gains, which scale with memory oversubscription pressure: with pipelined migration, <span id="S7.SS3.p5.6.1" class="ltx_text">MSched</span> achieves 1.27$\times$, 1.39$\times$, and 1.51$\times$ speedups under 150%, 200%, and 300% subscription, respectively. These results highlight the criticality of migration bandwidth for proactive scheduling. As links continue to scale, with PCIe bandwidth doubling per generation (PCIe 7.0$\times$<!-- -->16 with 256 GB/s$\times$<!-- -->2), and new fabrics like CXL \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">cxl-survey-csur24</span>\] and NVLink C2C (450 GB/s$\times$<!-- -->2) \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvlinkc2c</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">nvlinkc2c-benchmark</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">GH200-study-arxiv24</span>\] becoming available, <span id="S7.SS3.p5.6.2" class="ltx_text">MSched</span> will benefit proportionally, further improving practicality under aggressive memory oversubscription.

</div>

<figure id="S7.F10.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F10.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:212.5pt;">
<img src="x17.png" id="S7.F10.1.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="680" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F10.2" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:212.5pt;">
<img src="x18.png" id="S7.F10.2.g1" class="ltx_graphics ltx_centering ltx_img_square" width="830" height="680" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
</div>
<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 10: </span><em>Comparison of throughput under different oversubscribed volumes (a) and ratios (b) between RTX 5080 and RTX 3080.</em></figcaption>
</figure>

<div id="S7.SS3.p6" class="ltx_para ltx_noindent">

<span id="S7.SS3.p6.1.1" class="ltx_text ltx_font_bold">Hardware differences.</span> We evaluate <span id="S7.SS3.p6.1.2" class="ltx_text">MSched</span> on the two testbeds—RTX 5080 (16GB, PCIe 5.0) and RTX 3080 (10GB, PCIe 4.0)—to illustrate how HBM capacity and interconnect bandwidth affect end-to-end performance. Using the workloads from §<a href="https://arxiv.org/html/2512.24637v2#S7.SS1" class="ltx_ref" title="Effectiveness of Proactive Memory Scheduling ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7.1</span></a>, Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F10.fig1" class="ltx_ref" title="Figure 10 ‣ Ablation Study ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">10</span></a> compares task throughput under varying oversubscribed volumes and ratios. Under equal oversubscribed volume, RTX 5080 consistently outperforms RTX 3080, and the gap widens as volume increases. As the total migration volume is directly related to the absolute oversubscribed volume, this divergence stems primarily from the interconnect bandwidth disparity, matching the results in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F9.fig1" class="ltx_ref" title="Figure 9 ‣ Ablation Study ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">9</span></a>. Conversely, at equal oversubscription ratio, the two GPUs deliver similar throughput. The absolute oversubscribed volume is smaller for RTX 3080 due to its smaller HBM capacity, partially masking its bandwidth disadvantage.

</div>

</div>

<div id="S7.SS4" class="section ltx_subsection">

### Overhead Analysis

<figure id="S7.F11.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div id="S7.F11.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:429.3pt;">
<img src="x19.png" id="S7.F11.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="311" alt="Refer to caption" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 11: </span><em>Control-plane overhead (<span id="S7.F11.fig1.3.1.1" class="ltx_text ltx_font_typewriter">madvise</span>) of <span id="S7.F11.fig1.3.1.2" class="ltx_text ltx_font_upright">MSched</span> under different task counts and memory allocation ratios.</em></figcaption>
</figure>

<div id="S7.SS4.p1" class="ltx_para">

Beyond the data-plane cost of page migration,<span id="S7.SS4.p1.1.1" class="ltx_text">MSched</span> introduces control-plane overhead, primarily incurred by the synchronous <span id="S7.SS4.p1.1.2" class="ltx_text ltx_font_typewriter">madvise</span> calls of each task required to reorder the driver’s eviction list during context switching. Using the workload from §<a href="https://arxiv.org/html/2512.24637v2#S7.SS1" class="ltx_ref" title="Effectiveness of Proactive Memory Scheduling ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7.1</span></a>, we measure the aggregate latency of <span id="S7.SS4.p1.1.3" class="ltx_text ltx_font_typewriter">madvise</span>s during a single context switch under varying task counts and memory allocations. As shown in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F11.fig1" class="ltx_ref" title="Figure 11 ‣ Overhead Analysis ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">11</span></a>, the latency scales linearly with the number of tasks. Since these handcrafted tasks exhibit constant memory access volume per timeslice, the number of pages to <span id="S7.SS4.p1.1.4" class="ltx_text ltx_font_typewriter">madvise</span> per task remains constant, making the total overhead proportional to the task count. We also observe a slight latency increase under higher memory pressure, attributed to the slower page table lookups within the driver as the total number of allocated pages grows. Even so, within typical task-count ranges (tens of tasks or fewer), the overhead remains under 1 ms, which is negligible compared to the data migration latency.

</div>

</div>

<div id="S7.SS5" class="section ltx_subsection">

### Comparison with Existing Systems

<div id="S7.SS5.p1" class="ltx_para ltx_noindent">

<span id="S7.SS5.p1.1.1" class="ltx_text ltx_font_bold">Paging optimization for single task.</span> Existing OS-level optimizations for GPU demand paging target single-task execution. We compare <span id="S7.SS5.p1.1.2" class="ltx_text">MSched</span> against SUV \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">suv-micro24</span>\], a state-of-the-art system that employs compile-time static analysis to identify data hotness, guiding memory placement and prefetching. Because SUV relies on data structures of legacy GPU drivers incompatible with latest GPUs, we conduct this comparison on our RTX 3080 testbed. It is worth noting that SUV requires access to kernel source code, precluding support for common deep learning frameworks like PyTorch and llama.cpp which rely on closed-source kernel libraries (e.g., cuDNN, cuBLAS). Consequently, we employ the workloads described in §<a href="https://arxiv.org/html/2512.24637v2#S7.SS1" class="ltx_ref" title="Effectiveness of Proactive Memory Scheduling ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">7.1</span></a>.

</div>

<figure id="S7.F12.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div id="S7.F12.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:429.3pt;">
<img src="x20.png" id="S7.F12.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="342" alt="Refer to caption" />
</div>
<br />

<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 12: </span><em>Comparison of end-to-end throughput between <span id="S7.F12.fig1.3.1.1" class="ltx_text ltx_font_upright">MSched</span>, SUV [<span class="ltx_ref ltx_missing_citation ltx_ref_self">suv-micro24</span>], and UM under different memory allocations on RTX 3080.</em></figcaption>
</figure>

<div id="S7.SS5.p2" class="ltx_para">

As shown in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F12.fig1" class="ltx_ref" title="Figure 12 ‣ Comparison with Existing Systems ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">12</span></a>, SUV exhibits poor performance in multitasking workloads, even slightly worse than the native demand paging. In contrast, <span id="S7.SS5.p2.1.1" class="ltx_text">MSched</span> delivers substantial gains, achieving a 7.18$\times$ throughput improvement over SUV under 300% memory subscription. The gap stems from a <span id="S7.SS5.p2.1.2" class="ltx_text ltx_framed ltx_framed_underline">fundamental limitation</span> in single-task optimizations: they are oblivious to the drastic working set transitions induced by context switching. Without scheduling-aware coordination, per-task policies collide, causing severe migration conflicts and thrashing. <span id="S7.SS5.p2.1.3" class="ltx_text">MSched</span> avoids this by inferring the cross-task memory access order using schedule timeline and enforcing a globally optimal, scheduling-aligned page placement policy.

</div>

<div id="S7.SS5.p3" class="ltx_para ltx_noindent">

<span id="S7.SS5.p3.1.1" class="ltx_text ltx_font_bold">Compute-only scheduling.</span> We finally compare <span id="S7.SS5.p3.1.2" class="ltx_text">MSched</span> against XSched \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\], representing systems that only schedule GPU computing resources. XSched is a state-of-the-art preemptive GPU task scheduler and delegates memory to demand paging under oversubscription. Beyond the throughput-oriented RR policy used earlier, <span id="S7.SS5.p3.1.3" class="ltx_text">MSched</span> seamlessly supports other customized policies like the priority policy that targets latency. We emulate a typical cloud colocation scenario mixing Real-Time (RT) tasks which have strict SLOs and runs at high priority and Best-Effort (BE) tasks which improves GPU utilization at low priority and is preempted on RT arrivals. We configure two test cases: using ResNet152 inference as RT task, and either ResNet152 inference (testcase I) or training (testcase T) as BE task. We tune their batch sizes so that RT consumes 6GB memory and BE consumes 12GB.

</div>

<div id="S7.SS5.p4" class="ltx_para">

Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F13.fig1" class="ltx_ref" title="Figure 13 ‣ Comparison with Existing Systems ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">13</span></a> shows the results. For RT tasks, <span id="S7.SS5.p4.3.1" class="ltx_text">MSched</span> proactively restores the working set during context switching, eliminating cold-start page faults. Consequently, <span id="S7.SS5.p4.3.2" class="ltx_text">MSched</span> reduces the $P_{99}$ latency of RT tasks by 4.06$\times$ on average compared to XSched, improving service quality. Also, <span id="S7.SS5.p4.3.3" class="ltx_text">MSched</span>’s scheduling-aware OPT placement policy reduces data movement and boosts the throughput of BE tasks by 2.43$\times$ on average, thus increasing overall utilization of the GPU hardware. Notably, the speedup for training task is marginally lower than for inference (2.8%), primarily due to its intermittent command launching (§<a href="https://arxiv.org/html/2512.24637v2#S6.SS2" class="ltx_ref" title="Memory Manager ‣ Proactive Memory Scheduling ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">6.2</span></a>).

</div>

<figure id="S7.F13.fig1" class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle" style="width:433.6pt;">
<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F13.1" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:286.2pt;">
<img src="x21.png" id="S7.F13.1.g1" class="ltx_graphics ltx_centering ltx_img_landscape" width="830" height="581" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
<div class="ltx_flex_cell ltx_flex_size_1">
<div id="S7.F13.2" class="ltx_block ltx_figure_panel ltx_minipage ltx_align_middle" style="width:143.1pt;">
<img src="x22.png" id="S7.F13.2.g1" class="ltx_graphics ltx_centering ltx_img_portrait" width="830" height="1099" alt="Refer to caption" />
</div>
</div>
<div class="ltx_flex_break">

</div>
</div>
<br />

<figcaption><span class="ltx_tag ltx_tag_figure">Figure 13: </span><em>Comparison of (a) latency CDF of RT tasks and (b) throughput speedup of BE tasks between <span id="S7.F13.fig1.4.1.1" class="ltx_text ltx_font_upright">MSched</span> and XSched [<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>] on RTX 5080 when inference (I) or training (T) serves as the BE task.</em></figcaption>
</figure>

</div>

</div>

<div id="S8" class="section ltx_section">

## Related Work

<div id="S8.p1" class="ltx_para ltx_noindent">

<span id="S8.p1.1.1" class="ltx_text ltx_font_bold">GPU scheduling systems.</span> Prior research has extensively explored GPU compute multiplexing techniques. For example, EffiSha \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">effisha-ppopp17</span>\], FLEP \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">flep-asplos17</span>\], PipeSwitch \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">pipeswitch-osdi20</span>\], REEF \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">reef-osdi22</span>\], XSched \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">xsched-osdi25</span>\], and others \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">chimera-asplos15</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gcaps-ecrts24</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">tally-asplos25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">gpreempt-atc25</span>\] introduce various GPU preemption mechanisms to enable low-latency task switching. Meanwhile, approaches such as NVIDIA MPS \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-mps</span>\], MIG \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">nvidia-mig</span>\], libsmctrl \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">libsmctrl-rtas23</span>\], Baymax \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">baymax-asplos16</span>\], Paella \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">paella-sosp23</span>\], Orion \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">orion-eurosys24</span>\], SGDRC \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">sgdrc-ppopp25</span>\], and LithOS \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">lithos-sosp25</span>\] improve GPU utilization through spatial sharing and enforce resource isolation. Other systems like Clockwork \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">clockwork-osdi20</span>\], TGS \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">tgs-nsdi23</span>\], and Shepherd \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">shepherd-nsdi23</span>\] focus on GPU scheduling policies tailored for deep learning workloads. However, these compute-centric works either implicitly assume that the GPU HBM can accommodate the aggregate memory footprint of all concurrent tasks, or delegate memory oversubscription to the native demand paging mechanism which brings severe performance degradation. In contrast, <span id="S8.p1.1.2" class="ltx_text">MSched</span> addresses this by treating task working set as a first-class citizen of the GPU context and proactively schedules memory across concurrent tasks.

</div>

<div id="S8.p2" class="ltx_para ltx_noindent">

<span id="S8.p2.1.1" class="ltx_text ltx_font_bold">Demand paging optimizations for single GPU task.</span> Prior work optimized GPU demand paging (typically CUDA UM) under memory oversubscription by exploiting program access characteristics to guide page eviction and prefetching. Examples include SUV \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">suv-micro24</span>\], DeepUM \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">deepum-asplos23</span>\], Sentinel \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">sentinel-hpca21</span>\], SwapAdvisor \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">swapadvisor-asplos20</span>\], and others \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">snurhac-hpdc21</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">huvm-atc22</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">earlyadaptor-ispass23</span>\]. Others, such as Forest \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">forest-isca25</span>\], ETC \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">etc-asplos19</span>\], and SMC \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">smc-icpp24</span>\], redesign hardware to improve demand paging routine and introduce specialized units (e.g., memory compressors) to reduce page migration overhead. However, these systems are tailored for single-task execution and unaware of concurrency. Applying them in multitasking scenarios creates severe cross-task page conflicts and thrashing, as evidenced in Fig. <a href="https://arxiv.org/html/2512.24637v2#S7.F12.fig1" class="ltx_ref" title="Figure 12 ‣ Comparison with Existing Systems ‣ Evaluation ‣ Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling"><span class="ltx_text ltx_ref_tag">12</span></a>. In comparison, <span id="S8.p2.1.2" class="ltx_text">MSched</span> co-designs the scheduler and memory manager, leveraging the global task timeline to avoid inter-task conflicts and enforce optimal page placement policy.

</div>

<div id="S8.p3" class="ltx_para ltx_noindent">

<span id="S8.p3.1.1" class="ltx_text ltx_font_bold">In-application GPU memory swapping.</span> Prior work implemented numerous application-level swapping techniques to circumvent HBM constraints, offloading model parameters \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">clockwork-osdi20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">pipeswitch-osdi20</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">moelightning-asplos25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">sirius-atc25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">powerinfer-sosp24</span>\], KV cache \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">lmcache-arxiv25</span>\], and tensors \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">capuchin-asplos20</span>\] to host CPU DRAM. Others spill GPU data to disk \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">powerinfer2-arxiv24</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">llminflash-arxiv24</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">flexgen-arxiv23</span>\] or offload across network \[<span class="ltx_ref ltx_missing_citation ltx_ref_self">mooncake-fast25</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">kunserve-arxiv24</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">llumnix-osdi24</span>, <span class="ltx_ref ltx_missing_citation ltx_ref_self">blitzscale-osdi25</span>\]. Conversely, <span id="S8.p3.1.2" class="ltx_text">MSched</span> operates at OS-level and is compatible with these in-application swapping solutions, as <span id="S8.p3.1.3" class="ltx_text">MSched</span> can accurately identify the actually accessed memory during runtime.

</div>

</div>

<div id="S9" class="section ltx_section">

## Conclusion

<div id="S9.p1" class="ltx_para">

This paper presents <span id="S9.p1.1.1" class="ltx_text">MSched</span>, the first OS-level scheduler tailored for memory-oversubscribed GPU multitasking. By leveraging the memory predictability of GPU tasks, <span id="S9.p1.1.2" class="ltx_text">MSched</span> extends GPU context switching to include proactive working set scheduling. Our experiments demonstrate its effectiveness on varying workloads.

</div>

</div>

</div>

<div class="ltx_page_logo">

Generated on Fri Jan 2 15:58:38 2026 by <a href="http://dlmf.nist.gov/LaTeXML/" class="ltx_LaTeXML_logo"><span style="letter-spacing:-0.2em; margin-right:0.1em;">L<span class="ltx_font_smallcaps" style="position:relative; bottom:2.2pt;">a</span>T<span class="ltx_font_smallcaps" style="font-size:120%;position:relative; bottom:-0.2ex;">e</span></span><span style="font-size:90%; position:relative; bottom:-0.2ex;">XML</span><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==" alt="Mascot Sammy" /></a>

</div>

</div>
