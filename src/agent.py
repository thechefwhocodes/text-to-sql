"""
Agent logic for text-to-SQL conversion.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pydantic import ValidationError

from src.llm import LLM
from src.models import DEFAULT_MODEL
from src.tools import get_tool, tool_definations
from src.turns import Conversation, SystemTurn, ToolTurn, UserTurn, to_messages
from src.utils import get_ddl

SYSTEM_PROMPT = """
You are a SQL analyst answering questions about a SQLite database.
Only select the columns needed to answer the question.

Schema: {ddl}"""

# One call to write the SQL, up to two more to fix it, one to summarise the rows.
MAX_STEPS = 4

NO_ANSWER = "Sorry, I couldn't answer that one."


@dataclass
class Answer:
    """What the agent produced for a single question."""

    text: str
    sql: Optional[str]
    rows: Optional[pd.DataFrame]
    sql_attempts: int
    latency_s: float
    cost_usd: float


class Agent:
    """A text-to-SQL agent holding one conversation against one database."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        model: str = DEFAULT_MODEL,
        llm: Optional[LLM] = None,
    ):
        self.conn = conn
        self.model = model
        self.llm = llm or LLM()
        self.conversation: Conversation = [
            SystemTurn(SYSTEM_PROMPT.format(ddl=get_ddl(conn)))
        ]

    def ask(self, question: str) -> Answer:
        """Answer one question, appending every turn to the conversation."""
        self.conversation.append(UserTurn(question))

        text, sql, rows = NO_ANSWER, None, None
        attempts, latency_s, cost_usd = 0, 0.0, 0.0

        for _ in range(MAX_STEPS):
            turn = self.llm.chat( 
                to_messages(self.conversation),
                model=self.model,
                tools=tool_definations(),
                tool_choice="auto",
            )
            self.conversation.append(turn)
            latency_s += turn.latency_s
            cost_usd += turn.cost_usd

            if not turn.tool_calls:
                text = turn.content
                break

            tool_call = turn.tool_calls[0]
            try:
                tool = get_tool(tool_call.function.name)
                args = tool.parse_tool_args(tool_call.function.arguments)
            except (json.JSONDecodeError, ValidationError) as e:
                error = json.dumps({"error": f"Malformed tool call: {e}"})
                self.conversation.append(ToolTurn(tool_call_id=tool_call.id, content=error))
                continue

            result = tool.run(self.conn, args)
            attempts += 1

            if result.rows is not None:
                rows = result.rows
            sql = args.get("sql") # for reporting Answer

            self.conversation.append(ToolTurn(tool_call_id=tool_call.id, content=result.content))

        return Answer(
            text=text,
            sql=sql,
            rows=rows,
            sql_attempts=attempts,
            latency_s=latency_s,
            cost_usd=cost_usd,
        )
