"""Thin wrapper around the Fireworks chat completions API (OpenAI-compatible).

Swap models by passing a different key from src.models.MODELS — every call
made through a FireworksLLM instance is timed and costed, so you can compare
models with FireworksLLM.summary().
"""

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.models import DEFAULT_MODEL, get_model

load_dotenv()


@dataclass
class LLMResponse:
    content: str
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    raw: Any  # full OpenAI ChatCompletion, e.g. for response.choices[0].message.tool_calls


class FireworksLLM:
    """Fireworks chat client that tracks latency/cost/tokens for every call."""

    def __init__(self, api_key: str | None = None):
        self._client = OpenAI(
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        )

    def chat(self, messages: list[dict], model: str = DEFAULT_MODEL, **kwargs) -> LLMResponse:
        """Send a chat completion request. Extra kwargs (temperature, tools, ...) pass through."""
        model_config = get_model(model)

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=model_config.id,
            messages=messages,
            **kwargs,
        )
        latency_s = time.perf_counter() - start

        usage = response.usage
        cost_usd = (
            usage.prompt_tokens * model_config.input_price_per_million
            + usage.completion_tokens * model_config.output_price_per_million
        ) / 1_000_000

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            latency_s=latency_s,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=cost_usd,
            raw=response,
        )


def main() -> None:
    llm = FireworksLLM()
    messages = [{"role": "user", "content": "Say hello in Spanish"}]

    response = llm.chat(messages)
    print(response.content)
    print(
        f"model={response.model} latency={response.latency_s:.2f}s "
        f"tokens={response.total_tokens} cost=${response.cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()
