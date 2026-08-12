"""CLI entry point. Run with: uv run cli (or python -m src.cli)"""

from src.agent import Agent
from src.utils import load_db

BANNER = "Text-to-SQL CLI — ask a question about the database in plain English. Type 'exit' to quit."


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

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        answer = agent.ask(question)

        if answer.sql:
            print(f"\nSQL:\n{answer.sql}")
        if answer.rows is not None:
            print(f"\n{answer.rows.to_string(index=False)}")

        print(f"\n{answer.text}")

    conn.close()


if __name__ == "__main__":
    main()
