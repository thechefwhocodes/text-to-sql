"""
Turn types that make up a conversation.

A conversation is just a list of Turns — one entry per system prompt, user
message, model response, or tool result. Turn is the interface: every subclass
implements `to_message`, which renders it into the shape the chat API expects.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from openai.types.chat import ChatCompletion

from src.models import ModelConfig


USER_ROLE = "user"
TOOL_ROLE = "tool"
SYSTEM_ROLE = "system"
ASSISTANT_ROLE = "assistant"


Role = Literal[SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE, TOOL_ROLE]


class Turn(ABC):
    """Base type every turn implements. Don't instantiate directly."""

    role: ClassVar[Role]

    def to_message(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class SystemTurn(Turn):
    """The instructions and schema. Always the first turn in a conversation."""
    content: str
    role: ClassVar[Role] = SYSTEM_ROLE


@dataclass
class UserTurn(Turn):
    """A question typed by the user."""
    content: str
    role: ClassVar[Role] = USER_ROLE


@dataclass
class AgentTurn(Turn):
    raw_message: Any
    model: str
    latency_s: float
    total_tokens: int
    cost_usd: float
    role: ClassVar[Role] = ASSISTANT_ROLE

    @property
    def content(self) -> str | None:
        """The model's text. None when it answered with a tool call instead."""
        return self.raw_message.content

    @property
    def tool_calls(self) -> list | None:
        """The tool calls the model made, or None when it replied with text."""
        return self.raw_message.tool_calls

    def to_message(self) -> dict:
        return self.raw_message.model_dump(exclude_none=True) # role and content is already present inside raw_message

    @classmethod
    def from_completion(
        cls,
        completion: ChatCompletion,
        model: str,
        model_config: ModelConfig,
        latency_s: float,
    ) -> "AgentTurn":
        usage = completion.usage
        return cls(
            raw_message=completion.choices[0].message,
            model=model,
            latency_s=latency_s,
            total_tokens=usage.total_tokens,
            cost_usd=model_config.cost(usage.prompt_tokens, usage.completion_tokens),
        )


@dataclass
class ToolTurn(Turn):
    """The result of a tool call, reported back to the model."""

    tool_call_id: str
    content: str
    role: ClassVar[Role] = TOOL_ROLE

    def to_message(self) -> dict:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


Conversation = list[Turn]


def to_messages(conversation: Conversation) -> list:
    """Render a conversation into the messages list the LLM API expects."""
    return [turn.to_message() for turn in conversation]
