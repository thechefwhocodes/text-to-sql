"""
The customer's current prototype: one prompt, no schema, no tools.
"""

import sqlite3

from src.agent import Answer
from src.llm import LLM
from src.tools import RunSQLTool, run_sql
from src.turns import USER_ROLE

BASELINE_PROMPT = "Convert this question to SQL:\n{question}"


def extract_sql(text: str) -> str:
    """Pull the SQL out of a reply, which usually arrives in a markdown code block."""
    if "```" not in text:
        return text.strip()

    block = text.split("```")[1]
    if block.startswith("sql"):
        block = block[len("sql") :]
    return block.strip()


def ask_baseline(
    conn: sqlite3.Connection, llm: LLM, question: str, model: str
) -> Answer:
    """Run the customer's prompt, then execute whatever SQL comes back."""
    sql_tool = RunSQLTool()

    turn = llm.chat(
        [{"role": USER_ROLE, "content": BASELINE_PROMPT.format(question=question)}],
        model=model,
    )
    sql = extract_sql(turn.content)
    result = sql_tool.run_sql(conn, sql)

    return Answer(
        text=turn.content,
        sql=sql,
        rows=result.rows,
        sql_attempts=1,
        latency_s=turn.latency_s,
        cost_usd=turn.cost_usd,
    )
