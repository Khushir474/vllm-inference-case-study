import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from app.utils.prompts import QA_REPORT_SYSTEM_PROMPT, QA_REPORT_USER_TEMPLATE

# httpx's default connection pool caps at 100 concurrent connections — silently
# throttling any --concurrency above 100 at the client, not the server. Benchmark
# concurrency levels beyond 100 need a wider pool or the measurement reflects our
# own client queuing, not the backend's real capacity.
_HTTP_CLIENT = httpx.Client(limits=httpx.Limits(max_connections=500, max_keepalive_connections=500))


@dataclass
class GenerationResult:
    raw_text: str | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None


class OpenAICompatibleClient:
    """Thin wrapper around any OpenAI-compatible chat completions endpoint.

    Both the hosted API (Together) and self-hosted vLLM speak this same
    protocol, so this one class serves both backends — only base_url,
    api_key, and model differ between them.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=_HTTP_CLIENT)
        self.model = model

    def generate_qa_report(self, transcript_text: str) -> GenerationResult:
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QA_REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": QA_REPORT_USER_TEMPLATE.format(transcript_text=transcript_text)},
                ],
                temperature=0,
            )
        except Exception as e:
            return GenerationResult(
                raw_text=None,
                latency_s=time.monotonic() - start,
                prompt_tokens=0,
                completion_tokens=0,
                error=str(e),
            )

        latency_s = time.monotonic() - start
        usage = response.usage
        return GenerationResult(
            raw_text=response.choices[0].message.content,
            latency_s=latency_s,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            error=None,
        )
