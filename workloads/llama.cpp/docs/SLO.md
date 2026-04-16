Below is a quick‑and‑dirty cheat‑sheet for thinking about **service‑level objectives (SLOs) when you run a large‑language‑model (LLM) serving stack**.  Treat the numbers as ball‑park starting points—you’ll tighten them once you measure your own workload.

### 1. The core metrics engineers actually watch

| Category                  | Metric you usually pin an SLO to          | Why it matters                                                                                                   |
| ------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Availability**          | Uptime % over 28–30 days (e.g., ≥ 99.9 %) | Users hate 502s. Cloud providers routinely promise 99.9 %. ([OpenAI][1])                                         |
| **Latency**               | *Time‑to‑first‑token* (TTFT) at P95/P99   | Controls the “is it alive?” feel. Baseten shows < 200 ms TTFT is where chat starts to feel snappy ([Baseten][2]) |
|                           | *Total generation time* (P95)             | End‑to‑end wait for a full answer—matters for API chaining & batch work                                          |
| **Throughput**            | Output tokens / second (usually P50)      | Determines how fast replies stream; OpenAI’s “Scale Tier” advertises ≥ 80–100 tok/s ([OpenAI][1])                |
|                           | Requests per second (RPS)                 | Capacity planning, cost per request                                                                              |
| **Correctness / success** | Non‑error response rate (%)               | Counts timeouts, 5xx, safety blocks, etc.                                                                        |
| **Cost**                  | \$ per 1 k generated tokens               | Execs will ask. Track along with perf.                                                                           |

### 2. Typical target bands by use‑case

| Use‑case (what users see)           | TTFT (P95)         | Output tok/s (P50)        | Completion deadline     | Uptime target |
| ----------------------------------- | ------------------ | ------------------------- | ----------------------- | ------------- |
| **Interactive chat / IDE copilot**  | ≤ 0.2 s            | ≥ 30 tok/s ([Baseten][2]) | N/A                     | ≥ 99.9 %      |
| **Real‑time voice agent**           | ≤ 0.1 s            | ≥ 50 tok/s                | N/A                     | ≥ 99.95 %     |
| **Async micro‑batch API**           | ≤ 2 s              | ≥ 20 tok/s                | Whole answer ≤ 10 s     | ≥ 99.5 %      |
| **Offline batch (data prep, eval)** | ≤ 60 s queue entry | N/A                       | Job ≤ 1 h  ([arXiv][3]) | ≥ 99 %        |

The numbers above line up with:

* Academic workload studies that classify requests into **Interactive (20 s P99 TTFT), Batch‑1 (1 min), Batch‑2 (1 h)** ([arXiv][3])
* Commercial offerings that sell higher tiers with **99.9 % uptime plus ≥ 80–100 tok/s baseline throughput** ([OpenAI][1])

### 3. Picking the right percentile & window

* **Percentile**: 95th for general dashboards; 99th when you really care about long tails (e.g., trading systems).
* **Window**: common practice is rolling 1‑ or 5‑minute buckets and to evaluate SLO compliance over a 28‑day period (matches Google SRE guidelines).

### 4. Instrumentation tricks

* Grab TTFT by emitting a span at request receipt and another when you stream the first token.
* Compute tokens/s directly from the stream iterator—avoid extra per‑token hooks.
* Export Prometheus metrics; build burn‑rate alerts (e.g., “more than 2 % of requests breach the SLO over the last hour”).

### 5. How teams usually set their first SLO

1. **Measure baseline** with load‑test traces that mirror real token counts.
2. **Pick a stretch but attainable target** (often P95 lat ≈ 1.5× current median).
3. **Attach error‑budget‑based alerts** so you only wake up when the budget is being spent too fast.
4. Iterate—tighten once you have headroom or relax if the budget burns too quickly.

---

**TL;DR** For an interactive, consumer‑facing LLM service you’ll usually start with :

* **Availability** ≥ 99.9 %
* **P95 TTFT** ≤ 200 ms
* **P50 output rate** ≥ 30 tokens/s
* **Non‑error rate** ≥ 99.99 %

Everything else (batch jobs, eval pipelines, long‑context summarization) gets looser numbers, often expressed as “finish within X minutes at P99”.

Use real traffic and steady monitoring to tune from there.

[1]: https://openai.com/api-scale-tier/ "Scale Tier for API Customers | OpenAI"
[2]: https://www.baseten.co/blog/understanding-performance-benchmarks-for-llm-inference/ "Understanding performance benchmarks for LLM inference | Baseten Blog"
[3]: https://arxiv.org/html/2407.00047v2 "Queue Management for SLO-Oriented Large Language Model Serving"

