import argparse
import json
from pathlib import Path

from tqdm import tqdm

from app.clients.judge import AnthropicJudge
from app.config import settings


def run(raw_path: Path) -> Path:
    judge = AnthropicJudge(api_key=settings.anthropic_api_key, model=settings.judge_model)

    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]

    out_dir = Path(settings.results_dir) / "judged"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / raw_path.name

    with open(out_path, "w") as f:
        for row in tqdm(rows, desc="judging"):
            if row["json_valid"]:
                judgment = judge.score(row["transcript_text"], row["raw_output"])
                row["quality_score"] = judgment.score
                row["quality_rationale"] = judgment.rationale
                row["judge_error"] = judgment.error
            else:
                row["quality_score"] = None
                row["quality_rationale"] = None
                row["judge_error"] = "skipped: source row was not valid JSON"
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} judged rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path, help="path to a results/raw/*.jsonl file from run_benchmark.py")
    args = parser.parse_args()
    run(args.raw_path)


if __name__ == "__main__":
    main()
