import json
import sqlite3

import pytest
from pydantic import ValidationError

from src.tools import TextToSQLTool


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def tool():
    return TextToSQLTool()


def test_check_sql_allows_select(tool):
    tool._check_sql("SELECT * FROM items")


def test_check_sql_allows_with(tool):
    tool._check_sql("WITH x AS (SELECT 1) SELECT * FROM x")


def test_check_sql_rejects_multiple_statements(tool):
    with pytest.raises(ValueError):
        tool._check_sql("SELECT 1; SELECT 2")


def test_check_sql_rejects_non_select(tool):
    with pytest.raises(ValueError):
        tool._check_sql("DELETE FROM items")


def test_run_sql_returns_rows(tool, conn):
    result = tool.run_sql(conn, "SELECT * FROM items ORDER BY id")
    assert result.rows is not None
    assert result.rows.to_dict("records") == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
    ]


def test_run_sql_returns_error_content_for_write_statement(tool, conn):
    result = tool.run_sql(conn, "DROP TABLE items")
    assert result.rows is None
    assert "error" in json.loads(result.content)


def test_run_sql_returns_error_content_for_bad_syntax(tool, conn):
    result = tool.run_sql(conn, "SELECT * FROM not_a_table")
    assert result.rows is None
    assert "error" in json.loads(result.content)


def test_run_dispatches_sql_key_from_args(tool, conn):
    result = tool.run(conn, {"sql": "SELECT COUNT(*) as n FROM items"})
    assert result.rows.to_dict("records") == [{"n": 2}]


def test_parse_tool_args_returns_dict_for_valid_json(tool):
    assert tool.parse_tool_args('{"sql": "SELECT 1"}') == {"sql": "SELECT 1"}


def test_parse_tool_args_raises_on_invalid_json(tool):
    with pytest.raises(json.JSONDecodeError):
        tool.parse_tool_args("{sql: 'SELECT 1'}")


def test_parse_tool_args_raises_on_schema_mismatch(tool):
    with pytest.raises(ValidationError):
        tool.parse_tool_args('{"query": "SELECT 1"}')  # wrong key name
