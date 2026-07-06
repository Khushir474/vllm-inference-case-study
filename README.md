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
  aws_vllm_setup.md             # AWS GPU instance specs + step-by-step vLLM setup
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
  it's reachable. See `docs/aws_vllm_setup.md` for exact AWS instance specs
  (`g6.xlarge`, ~$0.805/hr), AMI, security group, and setup steps — including
  budget guidance, since the GPU instance is the only real cost risk in this
  study (Together AI + Anthropic judge costs stay under ~$10 even at 1,000
  transcripts).

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

Pipeline built, not yet run end-to-end. Current plan: pilot with 10
transcripts per backend to validate the pipeline, then scale up the sample
size once results look sane.
