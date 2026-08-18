import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.agent import Agent, Response
from src.utils import load_db

NUM_OF_RUNS = 1
QUESTION_WITH_ANSWERS_PATH = Path("data/dev_questions_with_answers.json")


def load_golden_answers(question_with_answers: list[dict]) -> dict[str, pd.DataFrame]:
    actual_answers = {}
    for qa in question_with_answers:
        actual_answers[qa["id"]] = pd.DataFrame(qa["expected_result"])

    return actual_answers


def is_correct(actual_rows: pd.DataFrame, expected_rows: pd.DataFrame):
    def to_string(value) -> str:
        is_number = isinstance(value, (int, float))
        return str(round(value, 2)) if is_number else str(value)

    def row_words(row) -> set[str]:
        words = set()
        for value in row:
            words.update(to_string(value).split())
        return words

    actual = [row_words(row) for row in actual_rows.itertuples(index=False)]
    expected = [row_words(row) for row in expected_rows.itertuples(index=False)]

    if len(actual) != len(expected):
        return False

    # find any actual row whose words are a superset of expected
    # row's words (extra columns like IDs are OK, missing ones aren't)
    # if no actual row (that isn't already claimed) covers the expected row, return False
    # else, claim the actual row so it can't be reused for another expected row
    for exp_words in expected:
        match = next((a for a in actual if exp_words <= a), None)
        if match is None:
            return False
        actual.remove(match)

    return True


def evaluate_agent(
    questions: dict,
    db_conn: sqlite3.Connection,
    actual_answers: dict[str, pd.DataFrame],
):
    report = defaultdict[str, int](int)
    for question in questions:
        print("------------------")
        print(question["question"])
        for _ in range(NUM_OF_RUNS):
            agent = Agent(conn=db_conn)
            response: Response = agent.ask(question["question"])
            if (
                response.text_to_sql_tool_turn
                and not response.text_to_sql_tool_turn.rows.empty
            ):
                rows = response.text_to_sql_tool_turn.rows
                if is_correct(rows, actual_answers[question["id"]]):
                    report[question["id"]] += 1


# def evaluate_baseline(questions: dict):


def main():
    db_conn: sqlite3.Connection = load_db()
    question_with_answers = json.loads(QUESTION_WITH_ANSWERS_PATH.read_text())
    actual_answers = load_golden_answers(question_with_answers)
    evaluate_agent(question_with_answers, db_conn, actual_answers)
    # baseline_results = evaluate_baseline(questions)


if __name__ == "__main__":
    main()
