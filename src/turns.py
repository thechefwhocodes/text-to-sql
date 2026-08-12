"""
Turn types that make up a conversation.

A conversation is just a list of Turns — one entry per user message, tool
call, or model response. Every turn carries a `content` field (the text that
becomes part of the prompt); turn-specific metadata (cost, tool args, ...)
rides alongside it but is stripped out by `to_messages()` before a
conversation is sent to the LLM.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypedDict

from openai.types.chat import ChatCompletion

from src.models import ModelConfig

Role = Literal["user", "assistant", "tool"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class Turn:
    """Base type. Don't instantiate directly — use one of the subclasses below."""

    content: str


@dataclass
class UserTurn(Turn):
    role: ClassVar[Role] = "user"


@dataclass
class ToolTurn(Turn):
    """content is the tool's response, formatted as text for the model to read."""

    tool_name: str
    tool_args: dict[str, Any]
    role: ClassVar[Role] = "tool"


@dataclass
class AgentTurn(Turn):
    """The LLM's own turn, plus the call stats we track it with."""

    model: str
    latency_s: float
    total_tokens: int
    cost_usd: float
    raw: Any = field(default=None, repr=False)  # full ChatCompletion, e.g. for message.tool_calls
    role: ClassVar[Role] = "assistant"

    @classmethod
    def from_completion(
        cls,
        completion: ChatCompletion,
        *,
        model: str,
        model_config: ModelConfig,
        latency_s: float,
    ) -> "AgentTurn":
        usage = completion.usage
        return cls(
            content=completion.choices[0].message.content,
            model=model,
            latency_s=latency_s,
            total_tokens=usage.total_tokens,
            cost_usd=model_config.cost(usage.prompt_tokens, usage.completion_tokens),
            raw=completion,
        )


Conversation = list[Turn]


def to_messages(conversation: Conversation) -> list[Message]:
    """Strip turn metadata down to the {role, content} shape the LLM API expects."""
    return [{"role": turn.role, "content": turn.content} for turn in conversation]
