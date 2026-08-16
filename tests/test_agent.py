import sqlite3
from dataclasses import dataclass
from typing import Optional

import pytest

from src.agent import Agent
from src.turns import AgentTurn


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    """Stands in for the SDK's ChatCompletionMessage — the only two things
    AgentTurn and to_message() actually read off it."""

    content: Optional[str] = None
    tool_calls: Optional[list] = None

    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}
        return {k: v for k, v in data.items() if not exclude_none or v is not None}


def fake_agent_turn(content: Optional[str] = None, tool_calls: Optional[list] = None) -> AgentTurn:
    return AgentTurn(
        raw_message=FakeMessage(content=content, tool_calls=tool_calls),
        model="fake-model",
        latency_s=0.0,
        cost_usd=0.0,
        total_tokens=0,
    )


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
        fake_agent_turn(tool_calls=[bad_call]),
        fake_agent_turn(content="Here is your answer."),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Here is your answer."
    assert answer.text_to_sql_tool_turn is None  # the malformed call never ran


def test_agent_recovers_from_tool_call_missing_required_field(conn):
    """Valid JSON that doesn't match the tool's schema (e.g. wrong key name)
    should also come back as a retryable error, not a crash."""
    bad_call = FakeToolCall("call_1", FakeFunction("run_sql", '{"query": "SELECT 1"}'))
    turns = [
        fake_agent_turn(tool_calls=[bad_call]),
        fake_agent_turn(content="Here is your answer."),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Here is your answer."
    assert answer.text_to_sql_tool_turn is None


def test_agent_runs_a_well_formed_tool_call(conn):
    good_call = FakeToolCall(
        "call_1", FakeFunction("run_sql", '{"sql": "SELECT * FROM items"}')
    )
    turns = [
        fake_agent_turn(tool_calls=[good_call]),
        fake_agent_turn(content="Found one row."),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Found one row."
    assert answer.text_to_sql_tool_turn.sql == "SELECT * FROM items"
    assert answer.text_to_sql_tool_turn.rows.to_dict("records") == [{"id": 1}]


def test_agent_recovers_from_sql_execution_error():
    """A query that fails to run (bad table name) comes back to the model as
    the tool's actual result content, not silently dropped — the model then
    fixes it and the retry succeeds."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (id INTEGER)")
    conn.execute("INSERT INTO items VALUES (1)")
    conn.commit()

    bad_sql_call = FakeToolCall(
        "call_1", FakeFunction("run_sql", '{"sql": "SELECT * FROM not_a_table"}')
    )
    good_sql_call = FakeToolCall(
        "call_2", FakeFunction("run_sql", '{"sql": "SELECT * FROM items"}')
    )
    turns = [
        fake_agent_turn(tool_calls=[bad_sql_call]),
        fake_agent_turn(tool_calls=[good_sql_call]),
        fake_agent_turn(content="Found one row after fixing the table name."),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("does not matter")

    assert answer.text == "Found one row after fixing the table name."
    assert answer.text_to_sql_tool_turn.sql == "SELECT * FROM items"
    assert answer.text_to_sql_tool_turn.rows.to_dict("records") == [{"id": 1}]

    # the failed attempt must have been sent back to the model as a real tool
    # result (not a message with no content), or the model could never have
    # known to retry
    messages = [t.to_message() for t in agent.conversation]
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert "error" in tool_messages[0]["content"]

    conn.close()


def test_agent_returns_no_tool_turn_when_the_question_needs_no_sql(conn):
    """A question the model answers directly, without calling the tool, must
    not crash and must not report a tool turn."""
    turns = [fake_agent_turn(content="I can't answer that from the database.")]
    agent = Agent(conn, llm=FakeLLM(turns))

    answer = agent.ask("why is the sky blue?")

    assert answer.text == "I can't answer that from the database."
    assert answer.text_to_sql_tool_turn is None


def test_agent_does_not_leak_tool_turn_across_questions(conn):
    """A follow-up question that doesn't touch the tool must not pick up the
    previous question's SQL/rows — each ask() call reports only its own turns."""
    good_call = FakeToolCall(
        "call_1", FakeFunction("run_sql", '{"sql": "SELECT * FROM items"}')
    )
    turns = [
        fake_agent_turn(tool_calls=[good_call]),
        fake_agent_turn(content="There is one row."),
        fake_agent_turn(content="I'm not sure, that's an opinion question."),
    ]
    agent = Agent(conn, llm=FakeLLM(turns))

    first = agent.ask("how many rows are there?")
    second = agent.ask("why do you think that is?")

    assert first.text_to_sql_tool_turn is not None
    assert second.text_to_sql_tool_turn is None
