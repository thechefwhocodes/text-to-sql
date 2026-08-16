import sqlite3
from dataclasses import dataclass
from typing import Optional

import pytest

from src.agent import Agent


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeTurn:
    content: Optional[str]
    tool_calls: Optional[list] = None
    latency_s: float = 0.0
    cost_usd: float = 0.0

    def to_message(self) -> dict:
        return {"role": "assistant", "content": self.content or ""}


class FakeLLM:
    """Returns each queued turn in order, one per call to `.chat`."""

    def __init__(self, turns):
        self.turns = iter(turns)

    def chat(self, messages, model=None, tools=None, tool_choice=None):
        return next(self.turns)


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (id INTEGER)")
    conn.execute("INSERT INTO items VALUES (1)")
    conn.commit()
    yield conn
    conn.close()


def test_agent_recovers_from_invalid_json_tool_call(conn):
    """Args that aren't valid JSON at all shouldn't crash the agent — the error
    goes back to the model as a tool result, same as a bad SQL query, and the
    agent keeps going on the next step."""
    bad_call = FakeToolCall("call_1", FakeFunction("run_sql", "{sql: 'SELECT 1'}"))
    turns = [
        FakeTurn(content=None, tool_calls=[bad_call]),
        FakeTurn(content="Here is your answer.", tool_calls=None),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Here is your answer."
    assert answer.sql_attempts == 0  # the malformed call never actually ran


def test_agent_recovers_from_tool_call_missing_required_field(conn):
    """Valid JSON that doesn't match the tool's schema (e.g. wrong key name)
    should also come back as a retryable error, not a crash."""
    bad_call = FakeToolCall("call_1", FakeFunction("run_sql", '{"query": "SELECT 1"}'))
    turns = [
        FakeTurn(content=None, tool_calls=[bad_call]),
        FakeTurn(content="Here is your answer.", tool_calls=None),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Here is your answer."
    assert answer.sql_attempts == 0


def test_agent_runs_a_well_formed_tool_call(conn):
    good_call = FakeToolCall(
        "call_1", FakeFunction("run_sql", '{"sql": "SELECT * FROM items"}')
    )
    turns = [
        FakeTurn(content=None, tool_calls=[good_call]),
        FakeTurn(content="Found one row.", tool_calls=None),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Found one row."
    assert answer.sql_attempts == 1
    assert answer.rows.to_dict("records") == [{"id": 1}]
