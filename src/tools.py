"""The tools the agent can call, and the code behind them.

Tool is the interface: a name, a schema (to_definition), and a body (run). The
schema and the code live on the same class so they can't drift apart, and the
agent looks a tool up by name rather than hardcoding which one to call — that's
what lets a second tool be added later without touching the agent loop.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.utils import query_db


@dataclass
class ToolResult:
    """What running a tool produced."""
    content: str
    rows: Optional[pd.DataFrame] = None


class Tool(ABC):
    """Base type every tool implements. Don't instantiate directly."""
    name: str
    defination: str
    parameters: BaseModel

    def to_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.defination,
                "parameters": self.parameters.model_json_schema(),
            },
        }
        

    @abstractmethod
    def run(self, conn: sqlite3.Connection, args: dict) -> ToolResult:
        """Execute the tool call and return its result."""

    def parse_tool_args(self, raw: str) -> dict:
        """Parse a tool call's JSON arguments"""
        data = json.loads(raw)
        return self.parameters.model_validate(data).model_dump()


class RunSQLArgs(BaseModel):
    """The arguments the model fills in to call run_sql."""
    sql: str = Field(description="A single read-only SQLite SELECT query.")


class RunSQLTool(Tool):
    name = "run_sql"
    defination = "Run a read-only SQL query against the database and return the rows. Use this for any question about the data."
    parameters = RunSQLArgs

    def run(self, conn: sqlite3.Connection, args: dict) -> ToolResult:
        return self.run_sql(conn, args["sql"]) # handle the case where sql is not in the args

    def _check_sql(self, sql: str) -> None:
        """Only support a single read-only statement"""

        stripped = sql.strip().rstrip(";").strip()

        if ";" in stripped:
            raise ValueError("Only a single SQL statement is allowed.")
        if not stripped.upper().startswith(("SELECT", "WITH")):
            raise ValueError("Only SELECT queries are supported.")


    def run_sql(self, conn: sqlite3.Connection, sql: str) -> ToolResult:
        """Execute the `sql`.

        Failures come back as a result, not an exception, so the model reads the
        error as the tool's output and gets a chance to correct its own query.
        """
        try:
            self._check_sql(sql)
            rows = query_db(conn, sql)
        except (ValueError, sqlite3.Error, pd.errors.DatabaseError) as e:
            return ToolResult(content=json.dumps({"error": str(e)}))

        content = json.dumps(
            {"row_count": len(rows), "rows": rows.to_dict("records")}, default=str
        )
        return ToolResult(content=content, rows=rows)


TOOLS: dict[str, Tool] = {tool.name: tool for tool in [RunSQLTool()]}


def get_tool(name: str) -> Tool:
    if name not in TOOLS:
        raise ValueError(
            f"Unknown tool '{name}'. Available tools: {', '.join(sorted(TOOLS.keys()))}"
        )
    return TOOLS[name]


def tool_definitions() -> list[dict]:
    return [tool.to_definition() for tool in TOOLS.values()]
