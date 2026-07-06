import argparse
from pathlib import Path

import pandas as pd

from app.benchmark.aggregate import load_judged_rows, summarize, to_markdown
from app.config import settings

METRIC_LABELS = {
    "latency_p50_s": ("median latency", "s", "lower"),
    "latency_p95_s": ("p95 latency", "s", "lower"),
    "throughput_tokens_per_s": ("throughput", "tokens/s", "higher"),
    "failure_rate": ("failure rate", "", "lower"),
    "json_validity_rate": ("JSON validity rate", "", "higher"),
    "mean_quality_score": ("mean judged quality", "/5", "higher"),
    "est_cost_per_request_usd": ("cost per request", "$", "lower"),
}


def build_narrative(summary: pd.DataFrame) -> str:
    by_backend = {row["backend"]: row for _, row in summary.iterrows()}
    if "hosted" not in by_backend or "vllm" not in by_backend:
        return (
            "Only one backend has results so far — run both `--backend hosted` and "
            "`--backend vllm` through the full pipeline to get a head-to-head comparison here."
        )

    hosted, vllm = by_backend["hosted"], by_backend["vllm"]
    lines = []
    for metric, (label, unit, better) in METRIC_LABELS.items():
        h, v = hosted[metric], vllm[metric]
        if pd.isna(h) or pd.isna(v):
            continue
        winner = "hosted" if (h < v) == (better == "lower") else "vllm"
        lines.append(f"- **{label}**: hosted {h:.4g}{unit} vs vLLM {v:.4g}{unit} — {winner} wins.")
    return "\n".join(lines)


def sample_rationales(df: pd.DataFrame, n: int = 3) -> str:
    sections = []
    for backend, g in df.groupby("backend"):
        judged = g.dropna(subset=["quality_score"]).head(n)
        if judged.empty:
            continue
        sections.append(f"### {backend}")
        for _, row in judged.iterrows():
            sections.append(
                f"- `{row['transcript_id']}` — score {row['quality_score']}/5: {row['quality_rationale']}"
            )
    return "\n".join(sections) if sections else "No judged rows yet."


def read_setup_notes(path: Path) -> str:
    if not path.exists():
        return "No setup notes found — see `docs/setup_notes_template.md`."
    return path.read_text()


def build_report(judged_dir: Path, setup_notes_path: Path) -> str:
    df = load_judged_rows(judged_dir)
    summary = summarize(df)

    return "\n\n".join(
        [
            "# Case Study: Hosted API vs Self-Hosted vLLM for Call-Transcript QA",
            "## Methodology",
            (
                "Both backends serve the same open-weight model (Llama-3.1-8B-Instruct) "
                "behind an OpenAI-compatible chat completions API — hosted via Together AI, "
                "self-hosted via vLLM on a cloud GPU — using an identical prompt and sample "
                "of transcripts from `data/92k-call-center-scripts/`. Quality is scored by an "
                "independent LLM judge (Claude) reading each generated QA report against its "
                "source transcript."
            ),
            "## Results",
            to_markdown(summary),
            "## Head-to-head",
            build_narrative(summary),
            "## Sample judge rationales",
            sample_rationales(df),
            "## Setup complexity",
            read_setup_notes(setup_notes_path),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged-dir", type=Path, default=Path(settings.results_dir) / "judged")
    parser.add_argument("--setup-notes", type=Path, default=Path("docs/setup_notes_template.md"))
    args = parser.parse_args()

    report = build_report(args.judged_dir, args.setup_notes)
    out_path = Path(settings.results_dir) / "case_study_report.md"
    out_path.write_text(report)
    print(report)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
