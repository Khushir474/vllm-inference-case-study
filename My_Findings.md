# Does vLLM Actually Matter? A Hands-On Benchmark of Hosted vs. Self-Hosted LLM Serving

*A case study in measuring what actually matters when you self-host an LLM; and what breaks along the way.*

## Quick orientation, before the jargon starts

**vLLM** is a serving engine: software that sits in front of a model and decides how to schedule incoming requests on the GPU, batching many together instead of running them one at a time. **"Naive" serving** is what you get without that: one request occupies the GPU until it's fully done, then the next one starts. **"Hosted"** means you're paying per API call to use someone else's GPU fleet instead of renting and running your own. Everything below is really just one question: how much does that batching software actually buy you, and when does it stop mattering?

## The question I set out to answer

"Just use vLLM" is common advice for anyone self-hosting an open-weights LLM. It's presented as an obvious win: continuous batching (folding new requests into work already running on the GPU, instead of waiting for a slot), PagedAttention (a memory-management trick that lets the GPU pack more concurrent requests into the same VRAM), better throughput. But *how much* better, under what conditions, and compared to what baseline? Vague claims like "vLLM is faster" don't hold up to scrutiny; faster than what, at what load, on what hardware?

I set out to answer this with a real task (not a synthetic benchmark), real infrastructure (rented cloud GPUs, not a pre-configured demo), and enough rigor to catch my own mistakes along the way; which, as it turned out, was most of the actual work.

The task: take a raw call-center transcript and generate a structured QA report from it (a realistic "LLM as a backend service" workload, not a toy chat demo). The comparison evolved through three distinct phases as I found and fixed confounds in my own methodology; that evolution is as much the story here as the final numbers.

## The dataset and task

I used the [AIxBlock 92k real-world call-center transcripts dataset](https://arxiv.org/abs/2507.02958); PII-redacted, real customer service calls with ASR confidence scores and word-level timestamps. For each transcript, the system prompts an LLM to extract a structured JSON QA report from the raw call text. This is a genuinely useful shape of workload: structured extraction from unstructured text, the kind of thing that shows up in real production LLM services.

Both backends I tested; a hosted API and a self-hosted server; sit behind the same OpenAI-compatible client code in my benchmark harness. Only the `base_url`, `api_key`, and `model` differ. That's a deliberate design choice: it means the *serving stack* is the only variable, not "did I write two different benchmarking codepaths that behave subtly differently."

## Phase 1: Hosted API vs. Self-Hosted vLLM: the comparison that looked done, but wasn't

The first comparison was the obvious one: Together AI's serverless hosted API (`Meta-Llama-3-8B-Instruct-Lite`, a quantized variant, $0.14 per million tokens) versus a self-hosted vLLM server running the full bf16 `Meta-Llama-3-8B-Instruct` on a rented RunPod GPU.

**A methodology bug caught early.** My first sweep used a different random sample of transcripts at each concurrency level. Transcript length varies a lot in this dataset, and that variance was confounding the throughput numbers; a "regression" I saw at one concurrency level turned out to just be a batch of longer transcripts, not a real performance change. The fix was straightforward once diagnosed: pin the same `--n --seed` across every level being compared, so `sample_transcripts()` (which is deterministic given those two values) returns the identical 200 transcripts every time.

**A second bug, this time in my own client.** `httpx`, the HTTP library underlying my client, defaults to a 100-connection pool. At concurrency=200, this was silently throttling the *hosted* backend; the API looked like it was hitting a ceiling, but it was actually my own client queuing connections locally rather than sending 200 requests in flight. I only caught this because the "ceiling" I saw didn't match my intuition about a well-resourced hosted API; a benchmark's own client code has to be validated as rigorously as the system under test, or you end up publishing a client-side artifact as a finding about someone else's infrastructure. Raising the pool to 500 connections fixed it; hosted throughput kept climbing cleanly past that point.

With both bugs fixed, here's what six concurrency levels (1 through 200, same 200-transcript sample at each) looked like on an L4 GPU. Read it as: hosted wins on raw speed, but watch the cost column; that's where the story flips.

| backend | concurrency | tokens/s | p50 latency | cost/request |
|---|---|---|---|---|
| hosted | 1 | 931 | 1.49s | $0.000211 |
| hosted | 200 | 35,652 | 5.37s | $0.000211 |
| vLLM (L4) | 1 | 184 | 8.37s | $0.000927 |
| vLLM (L4) | 200 | 2,569 | 67.09s | $0.000066 |

Hosted wins raw throughput by roughly 14x at concurrency=200; unsurprising, since Together is presumably running a much larger GPU fleet than my one rented card. But vLLM's *cost per request* stays competitive and pulls ahead from concurrency=10 onward, because hosted's per-token pricing is flat regardless of load, while a self-hosted GPU's fixed hourly cost gets amortized across more concurrent work as load increases. That's a real, defensible finding; self-hosting doesn't win on raw throughput against a bigger fleet, but it wins on cost-per-request once there's enough concurrent volume to keep the GPU busy.

vLLM also hit a hard ceiling past concurrency≈50 on the L4; confirmed directly via vLLM's own exposed KV-cache (KV-cache = the GPU memory holding each in-flight request's running context; more concurrent requests means more of it) metrics, not inferred. Marginal throughput gains dropped from +585% (c1→c10) to under +1% per step past c50. That's the GPU's VRAM running out of room for more concurrent KV-cache, not a software limit.

**A caveat I didn't smooth over**: vLLM's JSON output validity was materially worse than hosted's; around 20-28% vs. hosted's 92-93%, confirmed to be genuine early-EOS behavior (`finish_reason: "stop"` mid-JSON), not a `max_tokens` truncation bug. Full-precision self-hosted generation on this particular structured-output task was just less reliable at producing valid JSON than the hosted quantized variant. Worth stating plainly rather than only reporting the numbers that make the story cleaner.

## Phase 2: vLLM vs. Naive on the Same GPU: the fleet-size illusion

Here's the thing about the Phase 1 result: it's confounded. Hosted wins raw throughput not necessarily because Together's *serving engine* is better, but because they're running more GPUs than I am. Comparing "vLLM on one rented GPU" against "some unknown-sized hosted fleet" doesn't isolate the variable I actually cared about; vLLM's own architecture (continuous batching, PagedAttention) versus a naive serving approach.

So I built a second comparison, designed to actually isolate that variable: vLLM versus a **naive baseline**; plain `transformers.generate()` behind a global Python lock, same model weights, same GPU. The naive server (`app/naive_serve.py`) exposes just enough of the OpenAI chat-completions schema that the existing benchmark client works against it unmodified. The lock is the entire point: it forces strict request-at-a-time processing, which is what "no batching" looks like in practice; no continuous batching, no PagedAttention, no shared KV cache across requests.

Both vLLM and naive ran on an A100 80GB PCIe pod, `--n 200 --seed 0`. Same GPU, same model, only the serving software differs; watch how flat naive's throughput stays while vLLM's climbs:

| concurrency | vLLM tok/s | naive tok/s | vLLM p50 | naive p50 |
|---|---|---|---|---|
| 1 | 911 | 405 | 1.68s | 4.14s |
| 200 | 30,209 | 435 | 8.19s | 346.25s |

At concurrency=1, vLLM is already about 2.2x faster, even with nothing to batch yet. This isn't from CUDA graphs; a separate measurement in this same study found those gave zero benefit at c=1 (8.72s vs 8.55s, within noise), since they only accelerate *batched* decode. The c=1 gap is general implementation efficiency (kernel choice, attention implementation) rather than anything concurrency-dependent. But the real story is what happens under load: naive's throughput is essentially flat (405 → 435 tok/s from c=1 to c=200) because there is *nothing to batch*; one lock, one request at a time, regardless of how many requests arrive. vLLM's throughput, by contrast, climbs to 30,209 tok/s at the same concurrency. That's a **~70x gap**, and it's entirely attributable to the serving engine, because everything else; model, weights, GPU, prompt; was held constant.

The latency story is even starker. Naive's p50 at c=200 is 346 seconds; the 200th request in a naive server's queue has to wait for all 199 ahead of it to finish, one at a time, with nothing to absorb that queueing. vLLM's p50 at the same concurrency is 8.19 seconds. This is what "continuous batching" buys you in concrete terms: instead of a growing serial queue, incoming requests get folded into the GPU's existing batch, so wall-clock time per request doesn't scale linearly with how many other requests happen to be in flight.

### The measurement itself broke twice, and both failures were instructive

Getting this comparison to actually run cleanly took more debugging than the comparison itself.

**First break: a schema mismatch nobody warns you about.** I initially tried to reuse vLLM's own official benchmarking CLI (`vllm bench serve`) against the naive server, since it's the same tool that produces the numbers vLLM itself publishes, and it captures metrics (TTFT: how long until the first word comes back; TPOT: how long each word after that takes) my own client doesn't. Every single request failed instantly with HTTP 422. Tracing it down: `vllm bench serve`'s OpenAI-chat backend always sends message content as `[{"type": "text", "text": "..."}]`; OpenAI's newer content-parts format; and always requests `stream: true` with `max_completion_tokens` rather than `max_tokens`. My naive server's Pydantic model declared `content: str`, so it rejected every request as unprocessable before generation even started. I hadn't misconfigured a flag; I'd built a server that only accepted a request shape from an earlier version of the OpenAI API spec. Fixed by widening the model to accept `str | list[dict]`, extracting the text from either shape, and; since naive's `generate()` call blocks until the whole response is ready anyway; implementing minimal SSE streaming that emits the full text as a single chunk. That's not a hack; it's an honest representation of what "no token streaming" actually costs: naive's real time-to-first-token *is* its total generation time, because there's no partial output until the whole call returns.

**Second break: a proxy timeout that looked like a naive-server failure but wasn't.** Once the schema was fixed, I re-ran the benchmark against the pod's public URL (through RunPod's HTTP proxy) and got a wall-clock time suspiciously *faster* than expected; 183 seconds instead of the predicted ~800. Only 16 of 200 requests actually succeeded; the rest failed with 504 Gateway Timeout from nginx. The proxy has its own request timeout, and since naive serializes every request through one lock, anything queued behind more than a handful of others simply outlived the proxy's patience before it ever got a turn. Nginx returning 504 to the client doesn't cancel the request server-side, either; the naive server kept grinding through its backlog of accepted-but-abandoned requests for minutes after the "benchmark" had already reported a result. I had to kill and restart the server to clear that backlog, then re-run the whole thing through a direct SSH tunnel (bypassing the proxy entirely) to get requests that actually complete. The corrected run: all 200 requests succeeded, wall clock 783.6 seconds, matching the naive per-request rate I'd measured independently.

The lesson in both cases is the same one I'd already learned once with the `httpx` connection pool: **the measurement apparatus is part of the system under test.** A proxy timeout, a client-side connection limit, an OS file-descriptor cap; these all produce numbers that look exactly like a finding about the system you're trying to measure, and the only way to tell the difference is to go check.

### Extrapolating naive to higher load (without spending two hours proving something already provable)

I wanted to know what naive looks like at n=2000; comparable to where I'd later push vLLM; but naive's architecture makes this fully predictable without actually running it. Because every request serializes through one lock, per-request service time is invariant to how many requests are queued: from the n=200 run, 783.6s / 200 requests = **3.918 seconds per request**, a constant. Queueing is strict FIFO, so a request in position *k* of the queue waits *k × 3.918s* before its turn even starts. Two columns barely move (throughput, cost) while two climb linearly (latency); that split is the whole point of a fully serial architecture:

| metric | n=200 (measured) | n=2000 (projected) |
|---|---|---|
| wall clock | 783.6s | ~7,836s (~130.6 min) |
| p50 latency | 346.3s | ~3,918s (~65.3 min) |
| p95 latency | 744.2s | ~7,444s (~124.1 min) |
| throughput | 434.6 tok/s | ~434.6 tok/s (flat) |
| cost/request | $0.0013 | ~$0.0013 (flat) |

I want to be explicit that this table is **analytical, not measured**; I didn't spend two hours proving what the lock's own logic already guarantees. Throughput and cost-per-request don't move because nothing about the naive architecture changes with offered load; latency scales linearly because the queue does. This is a case where understanding *why* a system behaves the way it does is more valuable than brute-forcing a confirmation of it.

## Phase 3: Pushing vLLM to Its Real Breaking Point: 200 wasn't the ceiling

The Phase 2 comparison capped vLLM's GPU memory utilization at 0.4 (deliberately, to leave room for naive running alongside it) and only went up to concurrency=200; and at that point, vLLM's throughput was *still climbing*, no plateau in sight. That's a different picture from the L4 in Phase 1, which flattened hard past concurrency≈50. The A100's extra VRAM meant more room for concurrent KV-cache before hitting the same kind of ceiling (`kv_cache_max_concurrency` of 31.2 vs. the L4's 7.59, confirmed via vLLM's own exposed metrics).

So the natural next question: if 200 wasn't enough to find the ceiling, what is? I moved to a dedicated A100 SXM 80GB pod, raised GPU utilization to 0.9 and `--max-num-seqs` to 2048 (vLLM's scheduler defaults to ~256, which would impose an artificial ceiling well before the real GPU-capacity one), and switched tooling from my own client to `vllm bench serve`; the same benchmarking CLI the vLLM project itself uses, run directly on the pod against `localhost:8000` to eliminate any network-hop variables. It reports time-to-first-token (TTFT) and time-per-output-token (TPOT) natively, metrics my own harness never captured. Watch the last two columns; TPOT barely moves, TTFT explodes. That split is the whole finding:

| concurrency | completed | req/s | total tok/s | mean TTFT | mean TPOT |
|---|---|---|---|---|---|
| 200 | 200 | 8.89 | 12,800 | 9.58s | 89ms |
| 500 | 500 | 13.20 | 19,004 | 11.80s | 127ms |
| 1000 | 1000 | 15.16 | 21,820 | 19.08s | 147ms |
| 2000 | 2000 | 9.25 | 13,312 | 107.27s | 145ms |

There it is: throughput peaks at concurrency=1000 (21,820 tok/s) and then genuinely **collapses by 39%** at concurrency=2000 (13,312 tok/s). This isn't a client artifact; the giveaway is in the TPOT column, which barely moves (89-147ms mean across every level, p99 stays in a tight 144-175ms band too). Once a request is actually running, per-token generation speed is unchanged. What's collapsing is TTFT; mean time-to-first-token explodes from 19.1 seconds to 107.3 seconds between concurrency=1000 and 2000, a 5.6x jump. The most likely explanation: `--max-num-seqs 2048` puts concurrency=2000 at roughly 98% of the scheduler's own admission ceiling, so nearly every request queues for a long time just waiting to be *admitted* into the batch, even though the batch itself runs at normal speed once a request is in it. The bottleneck at extreme concurrency isn't compute; it's the scheduler's own admission control (the gatekeeping step that decides when a queued request is *allowed to start*, separate from how fast it runs once it does).

**One more false alarm caught before it became a wrong conclusion.** The first attempt at concurrency=2000 showed 995 of 2000 requests failing; a 49.75% failure rate that looked, on its face, exactly like "vLLM breaks under load." It wasn't. It was `ulimit -n` defaulting to 1024 open file descriptors on that pod, well under the ~2000 simultaneous connections the benchmark needed. Raising it to 65536 and re-running produced zero failures and the identical throughput-collapse pattern above. This is the same category of mistake as the `httpx` connection pool and the nginx proxy timeout, just one layer further down the stack; an OS resource limit masquerading as a finding about the server being tested. I'm flagging it explicitly here because it's exactly the kind of thing that's easy to report wrong under time pressure: "vLLM fails at 2000 concurrent requests" would have been a clean, confident, and *incorrect* headline.

## What this actually demonstrates

Pulling the three phases together, the honest picture is:

- **vLLM's advantage is a function of load, not a fixed multiplier.** At concurrency=1, it's meaningfully faster than naive (~2.2x) but nowhere near dramatic. By concurrency=200, the gap is ~70x. The technology's entire value proposition is invisible at low concurrency and dominant at high concurrency; which makes sense once you understand *what* continuous batching actually does (there's nothing to batch with one request in flight).
- **vLLM has a real, hardware-bound ceiling, and it's about memory, not compute.** The L4 saturated near concurrency=50; the A100, with far more VRAM, kept climbing to concurrency=1000 before its own scheduler-driven collapse at 2000. Bigger cards push the ceiling further out, but they don't eliminate it; and the way the ceiling manifests (a TTFT/admission problem, not a TPOT/compute problem) matters for how you'd actually tune around it in production (raising `max-num-seqs` further, or scaling horizontally, rather than assuming you need more raw compute).
- **Self-hosting doesn't have to win on raw throughput to be worth it.** Against Together's larger fleet, vLLM never wins on tokens/sec; but it wins on cost-per-request from moderate concurrency onward, and that's the metric that usually matters for a production service with predictable load.
- **The majority of the actual work here was catching my own instrumentation lying to me.** A confounded sample, a client-side connection pool, a proxy timeout, an OS file-descriptor limit; every one of these produced a plausible-looking "finding" about the system under test that was actually a bug in how I was measuring it. None of the headline numbers above are trustworthy without having gone and checked that the measurement apparatus itself wasn't the thing being measured.

If you're deciding whether to self-host an LLM instead of paying for a hosted API, the answer isn't "vLLM good, hosted bad"; it's: know your concurrency, and know your cost model. If your traffic is low and bursty, a hosted API is simpler and probably cheaper. If you're running enough concurrent volume to keep a GPU busy, self-hosting with a real batching engine gets cheaper per request than paying per token forever; but only past a certain load, and only if you've actually measured where that crossover is for your own workload, not assumed it.

## Tooling and infrastructure notes (the unglamorous half)

A few operational details worth recording, since they consumed real time and are the kind of thing that doesn't show up in a results table but absolutely shows up in how long a project like this takes:

- **RunPod's proxy has real limitations.** `scp`/`sftp` don't work through it at all (the SSH subsystem request fails outright); large file transfers over the PTY-based fallback (base64-encode, paste through an interactive shell) silently corrupt past a certain size due to terminal line-wrapping. The practical fix for pulling back multi-megabyte benchmark result files was extracting just the fields needed via a short Python one-liner executed over SSH, rather than moving the whole file; small, targeted extraction beats a flaky bulk transfer. (The correct *production* answer, which I noted but didn't build for a one-off study, is to have the compute node write results directly to object storage rather than relying on interactive file transfer at all.)
- **Some pod templates bake a specific start command into the container's PID 1.** One template I used had `vllm serve <model>` running as the container's actual entrypoint, not a background convenience process; attempting to `pkill` it and swap models by hand just triggered the whole container restarting with the same original command. The only way to actually change what it served was editing the pod's Container Start Command through the provider's console directly.
- **Dependency resolution across `torch`/`vllm`/`transformers` versions is fragile enough to break a working environment with one `pip install`.** Installing a recent `vllm` pulled in a `torch` build requiring a newer CUDA runtime than the pod's driver supported, breaking a server that had been working moments before. The fix was pinning an older `vllm` release compatible with the already-installed, driver-compatible `torch`, and; once I needed both a naive server and a `vllm bench serve` client coexisting on the same pod; isolating the second install into its own virtualenv so it couldn't touch the first one's dependencies again.

None of these are exotic problems. They're the ordinary friction of running real infrastructure instead of a curated demo, and getting through them is arguably a more representative signal of practical ML-infra skill than the final benchmark numbers are.

## Repo structure, for reference

```
app/
  config.py                    # settings for hosted/vLLM/naive backends, read from .env
  naive_serve.py                # naive baseline: transformers.generate() behind a global lock
  data_prep/load_transcripts.py # deterministic transcript sampling
  clients/openai_compatible.py  # one client for hosted API, vLLM, and naive
  benchmark/
    run_benchmark.py            # generates QA reports for a backend, writes raw/*.jsonl
    aggregate.py                 # summarizes runs into a comparison table
    vllm_metrics.py               # polls vLLM's own KV-cache/queue metrics during a run
docs/
  setup_notes.md                 # manual setup-friction log per backend
  runpod_vllm_setup.md            # RunPod GPU pod specs and vLLM setup steps
results/, results_naive_vs_vllm/  # raw generations, aggregated summaries, per-run experiment log
```

All results referenced above are reproducible from the raw per-request data checked into `results/` and `results_naive_vs_vllm/`, and every run is logged with its exact configuration in `experiment_log.jsonl`; nothing here is hand-typed from memory.
