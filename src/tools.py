"""The tools the agent can call, and the code behind them."""

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field

from src.turns import TableSchemaToolTurn, TextToSQLToolTurn, ToolTurn
from src.utils import get_ddl_for_table, query_db


class ToolResult(ABC):
    """What running a tool produced."""


class Tool(ABC):
    """Base type every tool implements. Don't instantiate directly."""

    name: str
    definition: str
    parameters: BaseModel

    def to_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.definition,
                "parameters": self.parameters.model_json_schema(),
            },
        }

    @abstractmethod
    def run(self, conn: sqlite3.Connection, args: dict) -> ToolResult:
        """Execute the tool call and return its result."""

    @abstractmethod
    def get_tool_turn(self, tool_call_id: str, result: ToolResult) -> ToolTurn:
        """Get the tool turn from the result."""

    def parse_tool_args(self, raw: str) -> dict:
        """Parse a tool call's JSON arguments"""
        data = json.loads(raw)
        return self.parameters.model_validate(data).model_dump()


@dataclass
class TableSchemaToolResult(ToolResult):
    ddl: str | None
    error: str | None = None


class TableSchemaToolArgs(BaseModel):
    """The arguments the model fills in to call run_sql."""

    table_name: str = Field(description="A single table name in the database.")


class TableSchemaTool(Tool):
    name: str = "get_table_schema"
    definition: str = "Run get_table_schema to fetch all the schema for a particual table available in the database"
    parameters: BaseModel = TableSchemaToolArgs

    def run(self, conn: sqlite3.Connection, args: dict) -> TableSchemaToolResult:
        return self.get_ddl_for_table(conn, args["table_name"])

    def get_tool_turn(
        self, tool_call_id: str, result: TableSchemaToolResult
    ) -> TableSchemaToolTurn:
        if result.error:
            return TableSchemaToolTurn(
                ddl=None,
                tool_call_id=tool_call_id,
                content=result.error,
            )

        return TableSchemaToolTurn(
            ddl=result.ddl,
            tool_call_id=tool_call_id,
            content=result.ddl,
        )

    def get_ddl_for_table(self, conn: sqlite3.Connection, table_name: str):
        try:
            ddl = get_ddl_for_table(conn, table_name)
            return TableSchemaToolResult(ddl)
        except Exception as e:
            return TableSchemaToolResult(error=json.dumps({"error": str(e)}), ddl=None)


@dataclass
class TextToSQLToolResult(ToolResult):
    sql: str
    error: str | None = None
    rows: pd.DataFrame | None = None


class TextToSQLToolArgs(BaseModel):
    """The arguments the model fills in to call run_sql."""

    sql: str = Field(description="A single read-only SQLite SELECT query.")


class TextToSQLTool(Tool):
    name = "run_sql"
    definition = "Run a read-only SQL query against the database and return the rows. Use this for any question about the data."
    parameters = TextToSQLToolArgs

    def run(self, conn: sqlite3.Connection, args: dict) -> TextToSQLToolResult:
        return self.run_sql(conn, args["sql"])

    def get_tool_turn(
        self, tool_call_id: str, result: TextToSQLToolResult
    ) -> TextToSQLToolTurn:
        if result.error:
            return TextToSQLToolTurn(
                sql=result.sql,
                rows=None,
                tool_call_id=tool_call_id,
                content=result.error,
            )

        return TextToSQLToolTurn(
            sql=result.sql,
            rows=result.rows,
            tool_call_id=tool_call_id,
            content=json.dumps(result.rows.to_dict("records"), default=str),
        )

    def _check_sql(self, sql: str) -> None:
        """Only support a single read-only statement"""

        stripped = sql.strip().rstrip(";").strip()

        if ";" in stripped:
            raise ValueError("Only a single SQL statement is allowed.")
        if not stripped.upper().startswith(("SELECT", "WITH")):
            raise ValueError("Only SELECT queries are supported.")

    def run_sql(self, conn: sqlite3.Connection, sql: str) -> TextToSQLToolResult:
        """Execute the `sql`.

        Failures come back as a result, not an exception, so the model reads the
        error as the tool's output and gets a chance to correct its own query.
        """
        try:
            self._check_sql(sql)
            rows = query_db(conn, sql)
        except (ValueError, sqlite3.Error, pd.errors.DatabaseError) as e:
            return TextToSQLToolResult(error=json.dumps({"error": str(e)}), sql=sql)

        return TextToSQLToolResult(sql=sql, rows=rows)


TOOLS: dict[str, Tool] = {
    tool.name: tool for tool in [TextToSQLTool(), TableSchemaTool()]
}


def get_tool(name: str) -> Tool:
    if name not in TOOLS:
        raise ValueError(
            f"Unknown tool '{name}'. Available tools: {', '.join(sorted(TOOLS.keys()))}"
        )
    return TOOLS[name]


def tool_definitions() -> list[dict]:
    return [tool.to_definition() for tool in TOOLS.values()]
