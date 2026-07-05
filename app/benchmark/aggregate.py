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
    return pd.DataFrame(rows)


def cost_per_request(row_backend: str, prompt_tokens: float, completion_tokens: float, mean_latency_s: float) -> float:
    if row_backend == "hosted":
        return (
            prompt_tokens * settings.hosted_price_per_input_token
            + completion_tokens * settings.hosted_price_per_output_token
        )
    # self-hosted: amortize GPU $/hr over the wall-clock time this request occupied the server
    return settings.vllm_gpu_hourly_cost_usd / 3600 * mean_latency_s


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for backend, g in df.groupby("backend"):
        n = len(g)
        latency = g["latency_s"]
        failure_rate = 1 - g["success"].mean()
        json_validity_rate = g["json_valid"].mean()
        quality_scores = g["quality_score"].dropna()
        mean_prompt_tokens = g["prompt_tokens"].mean()
        mean_completion_tokens = g["completion_tokens"].mean()
        total_tokens = (g["prompt_tokens"] + g["completion_tokens"]).sum()
        total_latency = latency.sum()

        summary.append(
            {
                "backend": backend,
                "model": g["model"].iloc[0],
                "n": n,
                "latency_p50_s": latency.median(),
                "latency_p95_s": latency.quantile(0.95),
                "throughput_tokens_per_s": total_tokens / total_latency if total_latency else float("nan"),
                "throughput_req_per_s": n / total_latency if total_latency else float("nan"),
                "failure_rate": failure_rate,
                "json_validity_rate": json_validity_rate,
                "mean_quality_score": quality_scores.mean() if len(quality_scores) else float("nan"),
                "est_cost_per_request_usd": cost_per_request(
                    backend, mean_prompt_tokens, mean_completion_tokens, latency.mean()
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
