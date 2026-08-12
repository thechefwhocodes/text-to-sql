"""
Catalog of Fireworks models available to src.llm.FireworksLLM.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    id: str  # full Fireworks model id, e.g. "accounts/fireworks/models/minimax-m3"
    input_price_per_million: float  # USD per 1M input tokens
    output_price_per_million: float  # USD per 1M output tokens

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost of a single call, given its token usage (assuming no thinking tokens)."""
        return (
            prompt_tokens * self.input_price_per_million
            + completion_tokens * self.output_price_per_million
        ) / 1_000_000


MODELS: dict[str, ModelConfig] = {
    "kimi-k2p7-code-fast": ModelConfig(
        id="accounts/fireworks/routers/kimi-k2p7-code-fast",
        input_price_per_million=1.90,
        output_price_per_million=8.00,
    ),
    "gpt-oss-120b": ModelConfig(
        id="accounts/fireworks/models/gpt-oss-120b",
        input_price_per_million=0.15,
        output_price_per_million=0.60,
    ),
}

DEFAULT_MODEL = "gpt-oss-120b"


def get_model(name: str) -> ModelConfig:
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {', '.join(sorted(MODELS))}"
        )
    return MODELS[name]
