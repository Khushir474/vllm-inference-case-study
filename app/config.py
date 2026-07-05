from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: str = "data/92k-call-center-scripts/home_ervice_inbound&telecom _outbound"

    # Hosted inference API (Together AI) — OpenAI-compatible.
    hosted_base_url: str = "https://api.together.xyz/v1"
    hosted_api_key: str = ""
    hosted_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # Together's published per-token price for this model (USD per token, not per 1K/1M).
    # Re-check https://www.together.ai/pricing before trusting this for real cost figures.
    hosted_price_per_input_token: float = 0.18 / 1_000_000
    hosted_price_per_output_token: float = 0.18 / 1_000_000

    # Self-hosted vLLM — OpenAI-compatible server, e.g. `vllm serve meta-llama/Llama-3.1-8B-Instruct`.
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "not-needed"
    vllm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Fill in once a GPU box is provisioned (e.g. RunPod A10G/L4 on-demand rate).
    vllm_gpu_hourly_cost_usd: float = 0.0

    # Judge model (Anthropic).
    anthropic_api_key: str = ""
    judge_model: str = "claude-sonnet-5"

    results_dir: str = "results"


settings = Settings()
