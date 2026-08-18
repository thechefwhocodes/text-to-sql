"""CLI entry point. Run with: uv run cli (or python -m src.cli)"""

from src.agent import Agent
from src.utils import load_db

BANNER = """
----------------
Text-to-SQL CLI — ask a question about the database in plain English. Type 'exit' to quit.
----------------
"""


def main() -> None:
    conn = load_db()
    agent = Agent(conn)

    print(BANNER)

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = agent.ask(question)

        if answer.text_to_sql_tool_turn:
            print(f"\nSQL:\n{answer.text_to_sql_tool_turn.sql}")
            print(
                f"---------\nRows:\n{answer.text_to_sql_tool_turn.rows.to_string(index=False)}"
            )

        print(f"---------\nAnswer:\n{answer.text}")

    conn.close()


if __name__ == "__main__":
    main()
