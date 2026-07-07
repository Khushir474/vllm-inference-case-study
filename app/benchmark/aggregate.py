import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from app.config import settings


def load_judged_rows(judged_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(judged_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"no judged result files found in {judged_dir}")
    df = pd.DataFrame(rows)
    if "quality_score" not in df.columns:
        # Rows straight from run_benchmark.py (not run through judge_quality.py yet)
        # don't have a quality_score column at all — treat as all-missing rather
        # than erroring, since quality judging is optional to this case study.
        df["quality_score"] = float("nan")
    return df


def cost_per_request(row_backend: str, prompt_tokens: float, completion_tokens: float, attributed_time_s: float) -> float:
    if row_backend == "hosted":
        return (
            prompt_tokens * settings.hosted_price_per_input_token
            + completion_tokens * settings.hosted_price_per_output_token
        )
    # Self-hosted: amortize GPU $/hr over the batch's wall-clock time divided by
    # request count, not mean per-request latency. Under concurrency, requests
    # overlap and share the same GPU-hour, so per-request latency (which grows
    # with queuing delay as concurrency rises) would double-count shared time and
    # make cost look like it gets *worse* under load — backwards from reality.
    return settings.vllm_gpu_hourly_cost_usd / 3600 * attributed_time_s


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if "concurrency" not in df.columns:
        df["concurrency"] = 1
    df["concurrency"] = df["concurrency"].fillna(1).astype(int)

    summary = []
    for (backend, concurrency), g in df.groupby(["backend", "concurrency"]):
        n = len(g)
        latency = g["latency_s"]
        failure_rate = 1 - g["success"].mean()
        json_validity_rate = g["json_valid"].mean()
        quality_scores = g["quality_score"].dropna()
        mean_prompt_tokens = g["prompt_tokens"].mean()
        mean_completion_tokens = g["completion_tokens"].mean()
        total_tokens = (g["prompt_tokens"] + g["completion_tokens"]).sum()

        # Under concurrency, requests overlap, so summing individual latencies
        # double-counts overlapping time — use the batch's actual wall-clock
        # time instead. Serial runs (concurrency=1) don't have this column, or
        # it happens to equal the latency sum anyway, so fall back to that.
        if "batch_wall_clock_s" in g.columns and g["batch_wall_clock_s"].notna().all():
            elapsed = g["batch_wall_clock_s"].iloc[0]
        else:
            elapsed = latency.sum()

        summary.append(
            {
                "backend": backend,
                "model": g["model"].iloc[0],
                "concurrency": concurrency,
                "n": n,
                "latency_p50_s": latency.median(),
                "latency_p95_s": latency.quantile(0.95),
                "throughput_tokens_per_s": total_tokens / elapsed if elapsed else float("nan"),
                "throughput_req_per_s": n / elapsed if elapsed else float("nan"),
                "failure_rate": failure_rate,
                "json_validity_rate": json_validity_rate,
                "mean_quality_score": quality_scores.mean() if len(quality_scores) else float("nan"),
                "est_cost_per_request_usd": cost_per_request(
                    backend, mean_prompt_tokens, mean_completion_tokens, elapsed / n
                ),
            }
        )
    return pd.DataFrame(summary)


def to_markdown(summary: pd.DataFrame) -> str:
    lines = ["# Benchmark Summary", "", summary.to_markdown(index=False, floatfmt=".4f")]
    lines += [
        "",
        "Setup complexity is not measured programmatically here — see "
        "`docs/setup_notes_template.md` for the qualitative log of each backend's setup effort.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judged-dir",
        type=Path,
        default=Path(settings.results_dir) / "judged",
        help="directory of *.jsonl files produced by judge_quality.py",
    )
    args = parser.parse_args()

    df = load_judged_rows(args.judged_dir)
    summary = summarize(df)

    out_path = Path(settings.results_dir) / "summary.md"
    out_path.write_text(to_markdown(summary))
    print(to_markdown(summary))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
