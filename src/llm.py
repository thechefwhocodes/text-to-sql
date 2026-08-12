"""Thin wrapper around the OpenAI-compatible chat completions API.

Swap models by passing a different key from src.models.MODELS — each config
carries its own base URL and API key, so the same code talks to Fireworks or
OpenAI. Every call is timed and turned into an AgentTurn, so latency/cost/tokens
travel with the conversation history automatically.
"""

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from src.models import DEFAULT_MODEL, get_model
from src.turns import AgentTurn

load_dotenv()


class LLM:
    """Fireworks chat client that turns every call into a timed, costed AgentTurn."""

    def chat(self, messages: list[dict], model: str = DEFAULT_MODEL, **kwargs) -> AgentTurn:
        """Send a chat completion request. Extra kwargs (temperature, tools, ...) pass through."""
        model_config = get_model(model)

        _client = OpenAI(
            api_key=os.environ.get(model_config.api_key_env),
            base_url=model_config.base_url,
        )

        start = time.perf_counter()
        completion = _client.chat.completions.create(
            model=model_config.id,
            messages=messages,
            **kwargs,
        )
        latency_s = time.perf_counter() - start

        return AgentTurn.from_completion(
            completion, model=model, model_config=model_config, latency_s=latency_s
        )
