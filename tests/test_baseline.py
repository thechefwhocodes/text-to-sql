import sqlite3
from dataclasses import dataclass

from src.baseline import ask_baseline, extract_sql


def test_extract_sql_from_plain_text():
    assert extract_sql("SELECT 1") == "SELECT 1"


def test_extract_sql_from_fenced_block_with_language_tag():
    text = "Here you go:\n```sql\nSELECT * FROM items\n```"
    assert extract_sql(text) == "SELECT * FROM items"


def test_extract_sql_from_fenced_block_without_language_tag():
    text = "```\nSELECT * FROM items\n```"
    assert extract_sql(text) == "SELECT * FROM items"


@dataclass
class FakeTurn:
    content: str
    latency_s: float = 0.0
    cost_usd: float = 0.0


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def chat(self, messages, model=None):
        return FakeTurn(content=self.content)


def test_ask_baseline_runs_the_extracted_sql_and_reports_a_tool_turn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (id INTEGER)")
    conn.execute("INSERT INTO items VALUES (1)")
    conn.commit()

    llm = FakeLLM("```sql\nSELECT * FROM items\n```")
    answer = ask_baseline(conn, "how many items?", llm=llm, model="fake-model")

    assert answer.text == "```sql\nSELECT * FROM items\n```"
    assert answer.text_to_sql_tool_turn.sql == "SELECT * FROM items"
    assert answer.text_to_sql_tool_turn.rows.to_dict("records") == [{"id": 1}]

    conn.close()


def test_ask_baseline_reports_a_tool_turn_even_when_the_sql_is_invalid():
    """The naive baseline prompt has no schema and no retries — a hallucinated
    table name should come back as a failed tool turn, not crash the eval."""
    conn = sqlite3.connect(":memory:")

    llm = FakeLLM("SELECT * FROM made_up_table")
    answer = ask_baseline(conn, "how many items?", llm=llm, model="fake-model")

    assert answer.text_to_sql_tool_turn is not None
    assert answer.text_to_sql_tool_turn.rows is None

    conn.close()
