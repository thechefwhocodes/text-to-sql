"""
Catalog of models available to src.llm.LLM.
"""

from dataclasses import dataclass

GPT_OSS_120B = "gpt-oss-120b"
GPT_5_4 = "gpt-5.4"

OPENAI_PROVIDER = "Open AI"
FIREWORKS_PROVIDER = "Fireworks AI"

OPENAI_KEY_ENV_VAR = "OPENAI_API_KEY"
FIREWORKS_API_KEY_ENV_VAR = "FIREWORKS_API_KEY"

OPENAI_BASE_URL = "https://api.openai.com/v1"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


@dataclass(frozen=True)
class ModelConfig:
    id: str  # full Fireworks model id, e.g. "accounts/fireworks/models/minimax-m3"
    base_url :str
    api_key_env: str
    provider: str
    input_price_per_million: float  # USD per 1M input tokens
    output_price_per_million: float  # USD per 1M output tokens

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost of a single call, given its token usage (assuming no thinking tokens)."""
        return (
            prompt_tokens * self.input_price_per_million
            + completion_tokens * self.output_price_per_million
        ) / 1_000_000


MODELS: dict[str, ModelConfig] = {
    GPT_OSS_120B: ModelConfig(
        id="accounts/fireworks/models/gpt-oss-120b",
        base_url=FIREWORKS_BASE_URL,
        api_key_env=FIREWORKS_API_KEY_ENV_VAR,
        provider=FIREWORKS_PROVIDER,
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    ),
    GPT_5_4: ModelConfig(
        id="gpt-5.4-2026-03-05",
        base_url=OPENAI_BASE_URL,
        api_key_env=OPENAI_KEY_ENV_VAR,
        provider=OPENAI_PROVIDER,
        input_price_per_million=2.5,
        output_price_per_million=15
    )
}

DEFAULT_MODEL = GPT_OSS_120B


def get_model(name: str) -> ModelConfig:
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {', '.join(sorted(MODELS.keys()))}"
        )
    return MODELS[name]
