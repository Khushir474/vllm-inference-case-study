import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from app.clients.openai_compatible import OpenAICompatibleClient
from app.config import settings
from app.data_prep.load_transcripts import sample_transcripts
from app.utils.schema import validate_json

BACKENDS = {
    "hosted": lambda: OpenAICompatibleClient(
        base_url=settings.hosted_base_url,
        api_key=settings.hosted_api_key,
        model=settings.hosted_model,
    ),
    "vllm": lambda: OpenAICompatibleClient(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
    ),
}


def run(backend: str, n: int, seed: int) -> Path:
    client = BACKENDS[backend]()
    transcripts = sample_transcripts(settings.data_dir, n, seed=seed)

    out_dir = Path(settings.results_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{backend}_{ts}.jsonl"

    model_name = settings.hosted_model if backend == "hosted" else settings.vllm_model

    with open(out_path, "w") as f:
        for t in tqdm(transcripts, desc=f"generating ({backend})"):
            result = client.generate_qa_report(t.text)
            success = result.error is None
            json_valid = False
            if success:
                json_valid, _ = validate_json(result.raw_text)

            row = {
                "transcript_id": t.id,
                "backend": backend,
                "model": model_name,
                "latency_s": result.latency_s,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "success": success,
                "json_valid": json_valid,
                "raw_output": result.raw_text,
                "error": result.error,
                "transcript_text": t.text,
            }
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(transcripts)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS), required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.backend, args.n, args.seed)


if __name__ == "__main__":
    main()
