from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: str = "data/92k-call-center-scripts"

    # Hosted inference API (Together AI) — OpenAI-compatible.
    # Llama-3.1-8B-Instruct-Turbo turned out to be dedicated-endpoint-only on
    # Together (confirmed via their own /v1/models catalog + matching community
    # reports) — not a billing-tier gate, it's just not served serverlessly
    # anymore. Meta-Llama-3-8B-Instruct-Lite is Together's actual serverless,
    # pay-per-token option, so both backends target Llama-3 (not 3.1) 8B.
    hosted_base_url: str = "https://api.together.xyz/v1"
    hosted_api_key: str = ""
    hosted_model: str = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    # Together's published per-token price for this model (USD per token, not per 1K/1M).
    # Re-check https://www.together.ai/pricing before trusting this for real cost figures.
    hosted_price_per_input_token: float = 0.14 / 1_000_000
    hosted_price_per_output_token: float = 0.14 / 1_000_000

    # Self-hosted vLLM — OpenAI-compatible server, e.g. `vllm serve meta-llama/Meta-Llama-3-8B-Instruct`.
    # "Lite" on the hosted side is Together's own quantized serving variant of
    # this same base checkpoint; self-hosting the full bf16 weights here is a
    # realistic real-world comparison (optimized hosted serving vs full-precision
    # self-hosting), not a mismatched model family.
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "not-needed"
    vllm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Fill in once a GPU box is provisioned (e.g. RunPod A10G/L4 on-demand rate).
    vllm_gpu_hourly_cost_usd: float = 0.0

    # Naive (non-vLLM) baseline — plain `transformers.generate()` behind a global
    # lock (app/naive_serve.py), same model/weights/GPU as vLLM above. Isolates
    # vLLM's continuous-batching/PagedAttention advantage from confounds like
    # GPU class or fleet size, which comparing against a hosted API can't do.
    naive_base_url: str = "http://localhost:8001/v1"
    naive_api_key: str = "not-needed"
    naive_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Judge model (Anthropic).
    anthropic_api_key: str = ""
    judge_model: str = "claude-sonnet-5"

    results_dir: str = "results"


settings = Settings()
