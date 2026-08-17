"""
The customer's current prototype: one prompt, no schema, no tools.
"""

import json
import sqlite3
from pathlib import Path

from src.agent import Response
from src.llm import LLM
from src.models import GPT_5_4
from src.tools import TextToSQLTool
from src.turns import USER_ROLE
from src.utils import load_db

QUESTIONS_PATH = Path("data/dev_questions_with_answers.json")
BASELINE_PROMPT = "Convert this question to SQL:\n{question}"


def extract_sql(text: str) -> str:
    """Pull the SQL out of a reply, which usually arrives in a markdown code block."""
    if "```" not in text:
        return text.strip()

    block = text.split("```")[1]
    block = block.removeprefix("sql")
    return block.strip()


def ask_baseline(
    conn: sqlite3.Connection, llm: LLM, question: str, model: str
) -> Response:
    """Run the customer's prompt, then execute whatever SQL comes back."""
    sql_tool = TextToSQLTool()

    turn = llm.chat(
        [{"role": USER_ROLE, "content": BASELINE_PROMPT.format(question=question)}],
        model=model,
    )
    sql = extract_sql(turn.content)
    result = sql_tool.run_sql(conn, sql)

    return Response(
        text=turn.content,
        latency_s=turn.latency_s,
        cost_usd=turn.cost_usd,
        text_to_sql_tool_turn=sql_tool.get_tool_turn("baseline", result),
    )


if __name__ == "__main__":
    llm = LLM()
    conn = load_db()
    questions = json.loads(QUESTIONS_PATH.read_text())
    response = ask_baseline(conn, llm, questions[1]["question"], model=GPT_5_4)
    print(response)
