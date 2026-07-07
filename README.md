# vLLM Case Study:  
## Hosted API vs Self-Hosted vLLM for Call-Transcript QA

A benchmark comparing a **hosted inference API** against a **self-hosted vLLM
server** on the same task: turning a call-center transcript into a structured
QA report. Both backends serve the same Llama-3 8B Instruct base checkpoint
behind an OpenAI-compatible API, so the app uses a single client for both —
only `base_url`/`api_key`/`model` differ. That's the whole point of vLLM
exposing an OpenAI-compatible server: it lets a real apples-to-apples
comparison exist without maintaining two client codepaths.

Note: Llama-3.1-8B-Instruct-Turbo was the original target model, but Together
AI turned out to have retired serverless (pay-per-token) access to it entirely
— it's dedicated-endpoint-only now, confirmed via their own `/v1/models`
catalog and matching reports from other users hitting the same error. Dedicated
endpoints bill hourly like a GPU rental, which defeats the point of a "hosted,
pay-per-token" comparison, so both backends target **Llama-3 (not 3.1) 8B
Instruct** instead — Together's `Meta-Llama-3-8B-Instruct-Lite` (their own
quantized serverless variant) on the hosted side, full bf16
`Meta-Llama-3-8B-Instruct` self-hosted on vLLM. This is arguably a more
realistic comparison anyway: optimized/quantized hosted serving vs
full-precision self-hosting is exactly the kind of tradeoff hosted APIs make
in production.

Metrics measured: **latency, throughput, cost, JSON validity, QA output
quality (LLM-judged), failure rate, and setup complexity**.

## Data

`data/92k-call-center-scripts/` is the
[AIxBlock 92k real-world call-center transcripts dataset](https://arxiv.org/abs/2507.02958)
(PII-redacted, word-level timestamps, ASR confidence scores). This study uses
`home_ervice_inbound&telecom _outbound/`, ~3,239 JSON transcripts. Only each
file's `text` field is used for QA generation.

## Architecture

```
app/
  config.py                 # settings for both backends + judge, read from .env
  data_prep/
    load_transcripts.py      # deterministic sampling of transcripts for a run
  utils/
    prompts.py                # QA report + judge prompt templates
    schema.py                  # QAReport pydantic model + JSON validation
  clients/
    openai_compatible.py      # one client, used for both hosted API and vLLM
    judge.py                   # Anthropic-based LLM judge for report quality
  benchmark/
    run_benchmark.py           # generates QA reports for a backend, writes results/raw/*.jsonl
    judge_quality.py           # scores a raw run's reports, writes results/judged/*.jsonl
    aggregate.py                # summarizes judged runs into results/summary.md
    build_report.py             # assembles a full write-up into results/case_study_report.md
docs/
  setup_notes_template.md      # manual log of setup friction per backend
  vllm_bench_serve.md          # optional supplementary raw-throughput check
  runpod_vllm_setup.md          # RunPod GPU pod specs + step-by-step vLLM setup
```

## Backends

- **Hosted**: [Together AI](https://www.together.ai/), OpenAI-compatible,
  serving `meta-llama/Meta-Llama-3-8B-Instruct-Lite` ($0.14/M tokens,
  confirmed serverless). Swap the provider/model in `app/config.py` if
  you'd rather use Fireworks/Groq/etc, or if Together's serverless catalog
  changes again — check `client.models.list()` against the target model
  before trusting it's actually servable serverlessly, the catalog listing
  something doesn't guarantee it.
- **Self-hosted**: vLLM serving `meta-llama/Meta-Llama-3-8B-Instruct` on a
  cloud GPU box you provision (this repo doesn't run vLLM locally — this
  Mac is arm64 with no CUDA GPU). Point `VLLM_BASE_URL` in `.env` at it once
  it's reachable. See `docs/runpod_vllm_setup.md` for exact RunPod GPU pod
  specs (~$0.39/hr L4, no quota/approval process), setup steps, and budget
  guidance — the GPU is the only real cost risk in this study (Together AI +
  Anthropic judge costs stay under ~$10 even at 1,000 transcripts). AWS was
  the original plan but got stuck behind a GPU vCPU quota request that was
  denied twice; RunPod has no equivalent approval gate.

## Setup

Requires Python 3.10+ (the code uses `X | None` type syntax). The system
`python3` on this Mac is 3.9 — use `python3.11`/`python3.12` if you have one
installed instead.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in HOSTED_API_KEY, ANTHROPIC_API_KEY, and (once provisioned) VLLM_BASE_URL / VLLM_GPU_HOURLY_COST_USD
```

## Running a benchmark pass

```bash
# 1. Generate QA reports for a backend against a sample of transcripts
python -m app.benchmark.run_benchmark --backend hosted --n 10
python -m app.benchmark.run_benchmark --backend vllm --n 10

# 2. Judge quality of each generated report against its source transcript
python -m app.benchmark.judge_quality results/raw/hosted_<timestamp>.jsonl
python -m app.benchmark.judge_quality results/raw/vllm_<timestamp>.jsonl

# 3. Aggregate all judged runs into a comparison table
python -m app.benchmark.aggregate

# 4. Build the full case-study write-up (results table + narrative + judge
#    rationale samples + setup notes) in one document
python -m app.benchmark.build_report
```

`results/summary.md` gets the per-backend table: latency (p50/p95),
throughput (tokens/s and req/s), estimated cost per request, JSON validity
rate, mean judged quality score, and failure rate. `results/case_study_report.md`
wraps that table in a methodology section, an auto-generated head-to-head
("hosted wins on X, vLLM wins on Y"), sample judge rationales, and the setup
notes.

`judge_quality.py` accepts `--judge-model` to override the default (Sonnet)
with a cheaper model (e.g. Haiku) for large-scale judging runs, saving
Sonnet for a final validation pass.

Setup complexity isn't computed — log it by hand per backend using
`docs/setup_notes_template.md`. See `docs/vllm_bench_serve.md` for an
optional supplementary raw-throughput check using vLLM's own benchmark CLI.

## Status

**The case study's focus is concurrent-request throughput and cost, not
serial per-request latency or output quality judging** — that's what vLLM's
continuous batching and PagedAttention are actually built to optimize, and
where self-hosting has a real story to tell. Quality judging (`judge_quality.py`)
is implemented but paused; it's not the point of this comparison.

Full results (`results/summary.md`), sampled across all 10 categories of the
~96k-transcript dataset. Every concurrency level below uses the **identical
200-transcript sample** (`--n 200 --seed 0`, fixed across every run for both
backends) — an earlier version of this sweep used a different random sample
per level, which confounded the throughput comparison with sample-length
variance; that flaw is fixed here.

| backend | concurrency | tokens/s | p50 latency | failure rate | cost/request |
|---|---|---|---|---|---|
| hosted (Together, Llama-3-8B-Instruct-Lite) | 1 | 931 | 1.49s | 2.0% | $0.000211 |
| hosted | 10 | 8,514 | 1.67s | 2.0% | $0.000211 |
| hosted | 25 | 4,944\* | 7.09s | 2.0% | $0.000211 |
| hosted | 50 | 23,481 | 2.25s | 2.0% | $0.000211 |
| hosted | 100 | 30,397 | 3.13s | 2.0% | $0.000211 |
| hosted | 200 | 35,652 | 5.37s | 2.0% | $0.000211 |
| vllm (self-hosted, Llama-3-8B-Instruct, fp8 KV cache) | 1 | 184 | 8.37s | 1.0% | $0.000927 |
| vllm | 10 | 1,262 | 12.02s | 1.0% | $0.000135 |
| vllm | 25 | 2,055 | 18.19s | 1.0% | $0.000083 |
| vllm | 50 | 2,522 | 28.37s | 1.0% | $0.000068 |
| vllm | 100 | 2,545 | 56.64s | 1.0% | $0.000067 |
| vllm | 200 | 2,569 | 67.09s | 1.0% | $0.000066 |

\* hosted's concurrency=25 throughput dip (lower than both c=10 and c=50) is
**real and reproducible** — it showed up independently in two separate
sweeps with different random samples. It isn't sample noise or a bug in our
client, but the underlying cause (something concurrency-25-specific on
Together's side — internal batching threshold, a rate-limit boundary) is
unconfirmed. Reported honestly as an open question rather than smoothed over.

**What changed vLLM's numbers from an earlier version of this sweep**: the
vLLM server was reconfigured to drop `--enforce-eager` (enabling CUDA graph
capture) and add `--kv-cache-dtype fp8` (roughly halving KV-cache memory per
token). This raised vLLM's own reported `kv_cache_max_concurrency` from 4.59
to 7.59 and its saturation plateau from ~1,900-2,100 tok/s to ~2,550 tok/s
(a genuine ~25-30% capacity increase) — confirmed via `vllm:num_requests_running`
roughly doubling at matching concurrency levels. CUDA graphs specifically
did *not* help serial (concurrency=1) throughput at all (8.37s vs the old
8.56s, within noise) — they only help *batched* decode, which only exists
under concurrency, as expected.

**A second, separate bug surfaced and was fixed mid-sweep**: the benchmark
client's HTTP library (`httpx`) defaults to a 100-connection pool, which was
silently throttling the *hosted* backend above concurrency=100 — the first
c=200 reading came back at 6.6 tok/s/req-equivalent (a drop from c=100),
which looked like hosted hitting a ceiling but was actually our own client
queuing connections locally. Raising the pool to 500 connections
(`app/clients/openai_compatible.py`) and re-running c=100/c=200 fixed this:
hosted's true c=200 throughput is 35,652 tok/s, continuing to climb rather
than drop. This is a reminder that a benchmark's own client code needs
validating as carefully as the systems it's measuring.

**Honest final takeaways:**
- **vLLM saturates hard past concurrency≈50 on a single L4 GPU** — this is
  a hardware capacity limit (confirmed directly via vLLM's own KV-cache
  metrics), not something further inference-engine tuning fixes. Marginal
  throughput gains collapse from +585% (c1→c10) to +63% (c10→c25) to +23%
  (c25→c50) to under +1% for every step after that.
- **Hosted shows no ceiling in the 1-200 range tested** (after fixing the
  connection-pool bug) — consistent with Together running a much larger
  GPU fleet than our single rented L4. At concurrency=200, hosted's raw
  throughput (35,652 tok/s) is **~14x higher** than vLLM's (2,569 tok/s).
  This is expected: one GPU has less aggregate capacity than many, and is
  not a knock on vLLM as a technology — it's the real tradeoff of
  self-hosting on constrained hardware vs. paying for a larger fleet.
- **Cost per request still favors self-hosted vLLM, and the advantage
  holds (even slightly improves) as concurrency rises.** At concurrency=1,
  vLLM is ~4.4x more expensive per request than hosted. From
  concurrency=10 onward, vLLM is consistently **~1.6-3.2x cheaper** per
  request (best at c=200: $0.000066 vs hosted's flat $0.000211 — hosted's
  cost is pure per-token pricing, insensitive to concurrency, while vLLM's
  fixed hourly GPU cost gets amortized across more and more concurrent
  work). **This is the legitimate, load-bearing result of this study**:
  self-hosting doesn't win on raw throughput against a bigger hosted fleet,
  but it wins decisively on cost-per-request once there's enough concurrent
  volume to keep the GPU busy.
- JSON validity for vLLM (Llama-3-8B-Instruct) is materially lower than
  hosted's quantized "Lite" variant (~20-25% vs ~92-93%) across all
  concurrency levels — confirmed as a genuine model behavior
  (`finish_reason: "stop"`, an early EOS token mid-JSON, not a
  truncation/config bug). Not the focus of this study, but a real caveat
  if structured-output reliability matters for a given use case.

Reproduce the concurrency comparison:
```bash
python -m app.benchmark.run_benchmark --backend vllm --n 200 --seed 0 --concurrency 50
python -m app.benchmark.run_benchmark --backend hosted --n 200 --seed 0 --concurrency 50
python -m app.benchmark.aggregate --judged-dir results/raw
```
(`--judged-dir results/raw` skips the quality-judging step and aggregates
straight from generation output, since judging is currently paused.) To
capture vLLM's own KV-cache/queue metrics around a run, see
`app/benchmark/vllm_metrics.py` (`snapshot` for a point-in-time read,
`poll` to sample continuously to a file during a benchmark run). **Always
use the same `--n`/`--seed` across every concurrency level you compare** —
`sample_transcripts()` is deterministic given those two values, and mixing
samples confounds the throughput comparison with transcript-length variance.

The vLLM side runs on a RunPod L4 pod (see `docs/runpod_vllm_setup.md`), one
gotcha not in that doc: this pod's template bakes `vllm serve Qwen/Qwen3-8B`
in as the container's own PID 1/start command (not just a suggestion you can
`pkill` around — killing it recreates the whole container). Swapping it to
serve Llama-3-8B-Instruct instead requires editing the pod's **Container
Start Command** in the RunPod console directly (args only, no `vllm serve`
prefix — the console adds that), and setting `HF_TOKEN` as a pod **environment
variable** rather than relying on an interactive `hf auth login`, since the
container gets recreated on each edit and non-env-var login state doesn't
survive that. The template also sets `VLLM_API_KEY` automatically — check
`env | grep VLLM_API_KEY` on the pod and pass it as a Bearer token.

## vLLM vs. Naive Serving (Same GPU) — Isolating vLLM's Real Advantage

The hosted-vs-vLLM comparison above is confounded by hardware: Together's
fleet is much larger than a single rented GPU, so it will always win on raw
throughput regardless of serving engine quality. To isolate **vLLM's actual
technology advantage** (continuous batching + PagedAttention), this section
compares vLLM against a naive baseline — plain `transformers.generate()`
behind a global lock (`app/naive_serve.py`), same model weights, **same
GPU**, so the only variable is the serving engine.

Moved to a bigger GPU for this comparison (RunPod **A100 80GB PCIe**,
~$1.19/hr) so vLLM and the naive server can run side-by-side with headroom;
results below are on this A100, not the L4 used in the hosted-comparison
section above — **not directly comparable to those numbers**, only to each
other. Results live in `results_naive_vs_vllm/` (separate from `results/` to
avoid mixing different-hardware vLLM runs).

vLLM on this A100 (`--gpu-memory-utilization 0.4` — deliberately capped
low to leave room for the naive server running alongside; see note below
about pushing this higher once naive testing is done), fixed `--n 200 --seed 0`
across all levels:

| concurrency | tokens/s | p50 latency | cost/request |
|---|---|---|---|
| 1 | 911 | 1.68s | $0.000572 |
| 10 | 5,754 | 2.69s | $0.000091 |
| 25 | 8,927 | 4.28s | $0.000058 |
| 50 | 12,264 | 6.07s | $0.000043 |
| 100 | 20,969 | 6.70s | $0.000025 |
| 200 | 30,209 | 8.19s | $0.000017 |

**No plateau found even at concurrency=200** — throughput keeps climbing
the whole way (unlike the L4, which flattened hard past concurrency≈50).
Consistent with `kv_cache_max_concurrency=31.2` on this A100 at only 0.4
GPU utilization (vs. 7.59 on the L4 at 0.9 utilization) — this card simply
has much more room before hitting the same KV-cache ceiling. Naive baseline
results and the true same-hardware vLLM-vs-naive comparison to follow once
that sweep completes (naive fully serializes every request via its lock, so
per RunPod-hour it's the more expensive, slower option almost by
construction — the interesting number is *how much* slower/more expensive,
not whether).

## vLLM Escalation to Its Real Breaking Point (`vllm bench serve`)

Our custom Python client (`ThreadPoolExecutor` + `httpx`) hits its own
scaling limits well before vLLM does — we'd already found and fixed one
(`httpx`'s default 100-connection pool silently throttling the *hosted*
backend earlier in this study). Rather than keep patching our own client,
this test switches to **vLLM's official benchmarking CLI** (`vllm bench
serve`), run directly on the pod against `localhost:8000` — no network hop,
no custom client code, the same tool the vLLM project and companies serving
on it use to publish real numbers. It reports **TTFT** (time to first
token) and **TPOT/ITL** (per-output-token and inter-token latency)
natively, metrics our own harness never captured.

Setup: fresh A100 SXM 80GB pod, `--gpu-memory-utilization 0.9
--kv-cache-dtype fp8 --max-num-seqs 2048` (raised from the default ~256 so
the scheduler itself doesn't impose an artificial ceiling before the real
GPU-capacity one does). Synthetic `random` dataset, 1300 input / 130 output
tokens per request (close to our real transcript averages), `n =
concurrency` at each level (full-burst), `--seed 0`, `--temperature 0`.

| concurrency | completed | req/s | output tok/s | total tok/s | TTFT mean/median/p99 (ms) | TPOT mean/median/p99 (ms) |
|---|---|---|---|---|---|---|
| 200 | 200 | 8.89 | 1,156 | 12,800 | 9,577 / 9,374 / 19,218 | 89 / 92 / 144 |
| 500 | 500 | 13.20 | 1,716 | 19,004 | 11,799 / 7,743 / 34,273 | 127 / 144 / 157 |
| 1000 | 1000 | 15.16 | 1,971 | 21,820 | 19,084 / 4,162 / 61,714 | 147 / 154 / 175 |
| 2000 | 2000 | 9.25 | 1,202 | 13,312 | 107,272 / 107,238 / 210,983 | 145 / 151 / 152 |

**Found the real breaking point.** Throughput peaks at **concurrency=1000**
(21,820 tok/s) and then genuinely **collapses by 39% at concurrency=2000**
(13,312 tok/s) — not a client artifact this time (see below), a real
performance cliff. Mean TTFT explodes from 19.1s to 107.3s (a 5.6x jump)
between those same two levels. Most likely cause: `--max-num-seqs 2048`
means concurrency=2000 sits right at ~98% of the scheduler's own admission
ceiling, so nearly every request queues for a long time waiting to even
start, even though — once admitted — per-token generation speed (TPOT/ITL)
barely changes across all four levels (145-175ms range throughout). The
degradation is almost entirely in *queueing/admission*, not in raw compute
once a request is actually running.

**Two real methodological catches along the way, both fixed:**
1. **The first concurrency=2000 attempt showed 995/2000 requests failing**
   (49.75% failure rate) — looked like a hard breaking point, but the
   actual cause was `ulimit -n` defaulting to **1024** open file
   descriptors on this pod, well under the 2000 simultaneous connections
   needed. Raised to 65536 (`ulimit -n 65536`) and re-ran: 0 failures, same
   throughput-collapse result as above. **This was a client/OS-limit
   artifact, not vLLM's real capacity ceiling** — worth stating plainly
   since it's the kind of false-negative that would have been easy to
   report as "vLLM breaks at 2000 concurrent requests" when the actual
   cause was completely unrelated to vLLM.
2. Every run uses `--save-result --save-detailed --result-dir
   /workspace/bench_results` on the pod, giving a full structured JSON per
   run (aggregate stats + per-request detail) rather than relying on
   parsed console output — `results_naive_vs_vllm/bench_serve/` holds the
   pulled-back subset of these (the largest detailed files stayed on the
   pod; aggregate fields were extracted via a direct `python3 -c` one-liner
   over SSH rather than transferring multi-MB JSON through the SSH
   proxy — see the data-pipeline note below).

**On moving data off the pod reliably**: `scp`/`sftp` don't work through
RunPod's SSH proxy (subsystem request fails), and pushing multi-MB files
through a `base64`-over-PTY-terminal round trip (our fallback for the
smaller `naive_serve.py` script transfer) gets unreliable at real file
sizes — terminal-width line wrapping corrupts or truncates large payloads.
For a one-off study this was fine to work around (extract just the
aggregate JSON fields via a short SSH-executed Python snippet instead of
moving the whole file). **The actual industrial answer is to have the pod
write results directly to object storage (S3 or equivalent)** rather than
relying on interactive SSH file transfer at all — this sidesteps the
proxy/PTY limitation entirely, and it's what a real pipeline would do
regardless of pod/proxy quirks: compute nodes push results to a durable
store, they don't get manually copied off after the fact.
