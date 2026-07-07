"""Automatic experiment logger — appends one structured record per benchmark
run to a persistent JSONL log, and renders that log into a markdown lab
notebook. This exists so every run's config and results are captured
consistently, rather than relying on manually writing up each one after the
fact (easy to forget a detail, or skip a run entirely under time pressure).
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from app.benchmark.aggregate import summarize
from app.benchmark.vllm_metrics import fetch_metrics

# vLLM Prometheus metric names worth pulling into every log entry — chosen
# because they explain *why* a run performed the way it did (queue time vs.
# actual compute time, cache pressure, preemption/recompute overhead), not
# just what the aggregate throughput/latency numbers were.
_VLLM_METRIC_KEYS = [
    "kv_cache_usage_perc",
    "kv_cache_max_concurrency",
    "num_requests_running",
    "num_requests_waiting",
    "num_preemptions_total",
    "request_queue_time_seconds_sum",
    "request_queue_time_seconds_count",
    "request_inference_time_seconds_sum",
    "request_inference_time_seconds_count",
    "time_to_first_token_seconds_sum",
    "time_to_first_token_seconds_count",
]


def _vllm_metrics_snapshot(base_url: str) -> dict | None:
    try:
        m = fetch_metrics(base_url)
    except Exception as e:
        return {"_error": str(e)}
    out = {k: m[k] for k in _VLLM_METRIC_KEYS if k in m}
    # Derive means from the Prometheus summary sum/count pairs — the raw
    # sums are cumulative across the server's whole lifetime, not just this
    # run, so these are lifetime averages, not per-run ones. Still useful as
    # a directional signal (e.g. is average queue time trending up).
    for stat in ("request_queue_time_seconds", "request_inference_time_seconds", "time_to_first_token_seconds"):
        s, c = out.get(f"{stat}_sum"), out.get(f"{stat}_count")
        if s is not None and c:
            out[f"{stat}_mean_lifetime"] = s / c
    return out


def log_run(
    raw_path: Path,
    backend: str,
    concurrency: int,
    n: int,
    seed: int,
    server_config: dict | None = None,
    vllm_metrics_url: str | None = None,
    log_path: Path | None = None,
) -> dict:
    """Append one record to the experiment log for a just-completed benchmark run."""
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["quality_score"] = df.get("quality_score", float("nan"))
    df["concurrency"] = concurrency
    df["batch_wall_clock_s"] = df.get("batch_wall_clock_s")
    summary = summarize(df).iloc[0].to_dict()

    latency = df["latency_s"]
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_path": str(raw_path),
        "backend": backend,
        "model": summary.get("model"),
        "concurrency": concurrency,
        "n": n,
        "seed": seed,
        "server_config": server_config or {},
        "throughput_tokens_per_s": summary.get("throughput_tokens_per_s"),
        "throughput_req_per_s": summary.get("throughput_req_per_s"),
        "latency_p50_s": summary.get("latency_p50_s"),
        "latency_p95_s": summary.get("latency_p95_s"),
        "latency_p99_s": latency.quantile(0.99),
        "failure_rate": summary.get("failure_rate"),
        "json_validity_rate": summary.get("json_validity_rate"),
        "est_cost_per_request_usd": summary.get("est_cost_per_request_usd"),
        "est_cost_per_million_tokens_usd": (
            summary.get("est_cost_per_request_usd", 0)
            / max((df["prompt_tokens"] + df["completion_tokens"]).mean(), 1)
            * 1_000_000
        ),
    }
    if vllm_metrics_url:
        record["vllm_metrics"] = _vllm_metrics_snapshot(vllm_metrics_url)

    log_path = log_path or (raw_path.parent.parent / "experiment_log.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def render_markdown(log_path: Path) -> str:
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    lines = ["# Experiment Log", ""]
    for r in records:
        cfg = ", ".join(f"{k}={v}" for k, v in r.get("server_config", {}).items())
        lines.append(f"## {r['timestamp']} — {r['backend']} concurrency={r['concurrency']} n={r['n']} seed={r['seed']}")
        if cfg:
            lines.append(f"Server config: `{cfg}`")
        lines.append("")
        lines.append(
            f"| tokens/s | req/s | p50 | p95 | p99 | failure rate | JSON valid | $/req | $/1M tok |\n"
            f"|---|---|---|---|---|---|---|---|---|\n"
            f"| {r['throughput_tokens_per_s']:.1f} | {r['throughput_req_per_s']:.3f} | "
            f"{r['latency_p50_s']:.2f}s | {r['latency_p95_s']:.2f}s | {r['latency_p99_s']:.2f}s | "
            f"{r['failure_rate']:.1%} | {r['json_validity_rate']:.1%} | "
            f"${r['est_cost_per_request_usd']:.6f} | ${r['est_cost_per_million_tokens_usd']:.2f} |"
        )
        vm = r.get("vllm_metrics")
        if vm and "_error" not in vm:
            lines.append("")
            lines.append(
                f"vLLM internals: KV cache {vm.get('kv_cache_usage_perc', 0):.1%} used, "
                f"max concurrency {vm.get('kv_cache_max_concurrency', 'n/a')}, "
                f"{vm.get('num_requests_running', 'n/a')} running / "
                f"{vm.get('num_requests_waiting', 'n/a')} waiting, "
                f"{vm.get('num_preemptions_total', 'n/a')} total preemptions (lifetime)"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    md = render_markdown(args.log_path)
    out = args.out or args.log_path.with_suffix(".md")
    out.write_text(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
