"""
Turn types that make up a conversation.

A conversation is just a list of Turns — one entry per system prompt, user
message, model response, or tool result. Turn is the interface: every subclass
implements `to_message`, which renders it into the shape the chat API expects.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd
from openai.types.chat import ChatCompletion

from src.models import ModelConfig

USER_ROLE = "user"
TOOL_ROLE = "tool"
SYSTEM_ROLE = "system"
ASSISTANT_ROLE = "assistant"


Role = Literal[SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE, TOOL_ROLE]


class Turn(ABC):
    """Base type every turn implements. Don't instantiate directly."""

    content: str | None
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
class ToolTurn(Turn):
    """The result of a tool call, reported back to the model."""

    content: str
    tool_call_id: str
    role: ClassVar[Role] = TOOL_ROLE

    def to_message(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }


@dataclass
class TextToSQLToolTurn(ToolTurn):
    """The result of a text to sql tool turn, reported back to the model."""

    sql: str
    rows: pd.DataFrame | None


@dataclass
class TableSchemaToolTurn(ToolTurn):
    """The result of a text to sql tool turn, reported back to the model."""

    ddl: str


@dataclass
class AgentTurn(Turn):
    model: str
    cost_usd: float
    raw_message: Any
    latency_s: float
    total_tokens: int
    role: ClassVar[Role] = ASSISTANT_ROLE

    @property
    def tool_calls(self) -> list | None:
        return self.raw_message.tool_calls

    @property
    def content(self) -> str | None:
        return None if self.tool_calls else self.raw_message.content

    def to_message(self) -> dict:
        if self.tool_calls:
            return {
                "role": self.role,
                "content": None,
                "tool_calls": self.raw_message.tool_calls,
            }

        return {"role": self.role, "content": self.content}

    @classmethod
    def from_completion(
        cls,
        model: str,
        model_config: ModelConfig,
        latency_s: float,
        completion: ChatCompletion,
    ) -> "AgentTurn":
        usage = completion.usage
        return cls(
            model=model,
            latency_s=latency_s,
            total_tokens=usage.total_tokens,
            raw_message=completion.choices[0].message,
            cost_usd=model_config.cost(usage.prompt_tokens, usage.completion_tokens),
        )


Conversation = list[Turn]


def to_messages(conversation: Conversation) -> list:
    """Render a conversation into the messages list the LLM API expects."""
    return [turn.to_message() for turn in conversation]
