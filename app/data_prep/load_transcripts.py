import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Transcript:
    id: str
    text: str
    audio_duration: float


def sample_transcripts(data_dir: str, n: int, seed: int = 0) -> list[Transcript]:
    """Deterministically sample n transcript JSON files from data_dir.

    Deterministic sampling (fixed seed) lets repeat runs against different
    backends operate on the exact same transcripts, which is required for
    the hosted-vs-vLLM quality/cost comparison to be apples-to-apples.
    """
    paths = sorted(Path(data_dir).rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no transcript JSON files found under {data_dir}")
    if n > len(paths):
        raise ValueError(f"requested {n} transcripts but only {len(paths)} available in {data_dir}")

    rng = random.Random(seed)
    chosen = rng.sample(paths, n)

    transcripts = []
    for path in chosen:
        with open(path) as f:
            data = json.load(f)
        transcripts.append(
            Transcript(
                id=path.stem,
                text=data["text"],
                audio_duration=data.get("audio_duration", 0),
            )
        )
    return transcripts
