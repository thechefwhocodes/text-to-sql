"""Thin wrapper around the Fireworks chat completions API (OpenAI-compatible).

Swap models by passing a different key from src.models.MODELS — every call
made through a FireworksLLM instance is timed and turned into an AgentTurn,
so latency/cost/tokens travel with the conversation history automatically.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from src.models import DEFAULT_MODEL, get_model
from src.turns import AgentTurn

load_dotenv()

FIREWORKS_API_KEY_ENV_VAR = "FIREWORKS_API_KEY"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksLLM:
    """Fireworks chat client that turns every call into a timed, costed AgentTurn."""

    def __init__(self, api_key: str | None = None):
        self._client = OpenAI(
            api_key=api_key or os.environ.get(FIREWORKS_API_KEY_ENV_VAR),
            base_url=FIREWORKS_BASE_URL,
        )

    def chat(self, messages: list[dict], model: str = DEFAULT_MODEL, **kwargs) -> AgentTurn:
        """Send a chat completion request. Extra kwargs (temperature, tools, ...) pass through."""
        model_config = get_model(model)

        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=model_config.id,
            messages=messages,
            **kwargs,
        )
        latency_s = time.perf_counter() - start

        return AgentTurn.from_completion(
            completion, model=model, model_config=model_config, latency_s=latency_s
        )


def main() -> None:
    llm = FireworksLLM()
    messages = [{"role": "user", "content": "Say hello in Spanish"}]

    turn = llm.chat(messages)
    print(turn.content)
    print(
        f"model={turn.model} latency={turn.latency_s:.2f}s "
        f"tokens={turn.total_tokens} cost=${turn.cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()
