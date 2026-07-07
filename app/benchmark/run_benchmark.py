import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from app.benchmark.experiment_log import log_run
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
    "naive": lambda: OpenAICompatibleClient(
        base_url=settings.naive_base_url,
        api_key=settings.naive_api_key,
        model=settings.naive_model,
    ),
}


def _build_row(backend: str, model_name: str, t, result) -> dict:
    success = result.error is None
    json_valid = False
    if success:
        json_valid, _ = validate_json(result.raw_text)
    return {
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


def run(backend: str, n: int, seed: int, concurrency: int = 1, server_config: dict | None = None) -> Path:
    client = BACKENDS[backend]()
    transcripts = sample_transcripts(settings.data_dir, n, seed=seed)

    out_dir = Path(settings.results_dir) / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_c{concurrency}" if concurrency > 1 else ""
    out_path = out_dir / f"{backend}{suffix}_{ts}.jsonl"

    model_name = {
        "hosted": settings.hosted_model,
        "vllm": settings.vllm_model,
        "naive": settings.naive_model,
    }[backend]

    batch_start = time.monotonic()
    if concurrency <= 1:
        rows = []
        for t in tqdm(transcripts, desc=f"generating ({backend})"):
            result = client.generate_qa_report(t.text)
            rows.append(_build_row(backend, model_name, t, result))
    else:
        # Continuous batching / paged attention (vLLM's throughput advantages) only
        # show up under concurrent load — firing requests one at a time never gives
        # the server more than one request to batch. This mode fires `concurrency`
        # requests at once via a thread pool (network I/O releases the GIL) so we can
        # measure real aggregate throughput instead of only serial per-request latency.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(
                tqdm(
                    pool.map(lambda t: client.generate_qa_report(t.text), transcripts),
                    total=len(transcripts),
                    desc=f"generating ({backend}, concurrency={concurrency})",
                )
            )
        rows = [_build_row(backend, model_name, t, r) for t, r in zip(transcripts, results)]
    batch_wall_clock_s = time.monotonic() - batch_start

    with open(out_path, "w") as f:
        for row in rows:
            row["concurrency"] = concurrency
            row["batch_wall_clock_s"] = batch_wall_clock_s
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} rows to {out_path} (concurrency={concurrency}, wall clock={batch_wall_clock_s:.1f}s)")

    # Auto-log this run so every experiment's config/results is captured
    # consistently in one place, rather than depending on someone remembering
    # to write it up afterward. vLLM-only metrics (KV cache, queueing) get
    # pulled in automatically when the backend exposes a /metrics endpoint.
    vllm_metrics_url = settings.vllm_base_url if backend == "vllm" else None
    try:
        log_run(out_path, backend, concurrency, n, seed, server_config=server_config, vllm_metrics_url=vllm_metrics_url)
    except Exception as e:
        print(f"warning: experiment logging failed ({e}), continuing anyway")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS), required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="number of requests to fire concurrently (>1 exercises vLLM's continuous batching)",
    )
    parser.add_argument(
        "--server-config",
        type=str,
        default=None,
        help='JSON string of server config to attach to the experiment log, e.g. \'{"gpu_memory_utilization": 0.9}\'',
    )
    args = parser.parse_args()
    server_config = json.loads(args.server_config) if args.server_config else None
    run(args.backend, args.n, args.seed, concurrency=args.concurrency, server_config=server_config)


if __name__ == "__main__":
    main()
