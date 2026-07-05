# Supplementary throughput check: `vllm bench serve`

The Python benchmark in `app/benchmark/` measures paired, quality-aware metrics
(latency, cost, JSON validity, judged quality, failure rate) per transcript, run
sequentially so results line up 1:1 across backends.

That harness does **not** stress-test raw serving throughput under concurrent
load. For a pure infra throughput number on the self-hosted vLLM box, run
vLLM's own benchmark CLI directly against it once the server is up:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://<vllm-host>:8000 \
  --endpoint /v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 200 \
  --request-rate inf
```

This reports request throughput, output token throughput, and latency
percentiles under load — a separate, complementary data point to the
per-request numbers in `results/summary.md`. It only applies to the
self-hosted side; hosted providers don't expose a comparable load-test target
(and running one against a shared multi-tenant API would just be hammering
someone else's infra).
