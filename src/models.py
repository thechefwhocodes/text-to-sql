"""Catalog of Fireworks models available to src.llm.FireworksLLM.

To try a new model: look up its id on https://fireworks.ai/models and its
$/1M token price on https://docs.fireworks.ai/serverless/pricing, then add
one ModelConfig entry to MODELS below. Nothing else needs to change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    id: str  # full Fireworks model id, e.g. "accounts/fireworks/models/minimax-m3"
    input_price_per_million: float  # USD per 1M input tokens
    output_price_per_million: float  # USD per 1M output tokens


MODELS: dict[str, ModelConfig] = {
    "minimax-m3": ModelConfig(
        id="accounts/fireworks/models/minimax-m3",
        input_price_per_million=0.30,
        output_price_per_million=1.20,
    ),
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
    """Look up a model by its short key (e.g. "minimax-m3")."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {', '.join(sorted(MODELS))}"
        ) from None
