"""CLI entry point. Run with: uv run cli (or python -m src.cli)"""


from src.utils import get_schema, load_db, print_table_schema


def main() -> None:
    # TODO: implement your interactive CLI here
    print("Hello from the Text-to-SQL CLI!")
    print("Implement your CLI in src/cli.py")

    db_connection = load_db()
    schemas = get_schema(db_connection)
    tables = schemas.keys()

    for table in tables:
        print_table_schema(db_connection, table)


if __name__ == "__main__":
    main()
