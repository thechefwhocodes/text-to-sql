import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.agent import Agent, Response
from src.baseline import ask_baseline
from src.models import DEFAULT_MODEL, GPT_5_4
from src.utils import load_db

NUM_OF_RUNS = 3
QUESTION_WITH_ANSWERS_PATH = Path("data/dev_questions_with_answers.json")


def load_expected_answers(question_with_answers: list[dict]) -> dict[str, pd.DataFrame]:
    expected_answers = {}
    for qa in question_with_answers:
        expected_answers[qa["id"]] = pd.DataFrame(qa["expected_result"])

    return expected_answers


def is_matching(actual_rows: pd.DataFrame, expected_rows: pd.DataFrame):
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


def print_report(report: dict[str, dict], question_with_answers: list[dict]):
    total_correct_responses, total_latency, total_cost = 0.0, 0.0, 0.0
    missed = []
    for question_id, stats in report.items():
        if stats["correct_response"] != NUM_OF_RUNS:
            missed.append(
                f"{question_id} ({int(stats['correct_response'])}/{NUM_OF_RUNS})"
            )

        total_correct_responses += stats["correct_response"]
        total_latency += stats["total_latency"]
        total_cost += stats["total_cost"]

    accuracy = round(
        (total_correct_responses / (NUM_OF_RUNS * len(question_with_answers))) * 100, 2
    )
    average_latency = round(
        total_latency / (NUM_OF_RUNS * len(question_with_answers)), 3
    )
    average_cost = round(total_cost / (NUM_OF_RUNS * len(question_with_answers)), 6)

    print(f"   accuracy: {accuracy}%")
    print(f"   latency : {average_latency}s/query")
    print(
        f"   cost    : ${average_cost}/query --> ${round(average_cost * 30_000, 2)}/day at 30,000 queries"
    )
    if missed:
        print(f"   missed  : {missed}")
    print("-------------------")


def evaluate_agent(
    db_conn: sqlite3.Connection,
    question_with_answers: list[dict],
    expected_answers: dict[str, pd.DataFrame],
):
    print(
        f"Evaluating Agent ({DEFAULT_MODEL}): {NUM_OF_RUNS} runs X {len(question_with_answers)} questions"
    )
    ask_fn = lambda q: Agent(conn=db_conn).ask(q)
    evaluate(question_with_answers, expected_answers, ask_fn)


def evaluate_baseline(
    db_conn: sqlite3.Connection,
    question_with_answers: list[dict],
    expected_answers: dict[str, pd.DataFrame],
):
    print(
        f"Evaluating Baseline ({GPT_5_4}): {NUM_OF_RUNS} runs X {len(question_with_answers)} questions"
    )
    ask_fn = lambda q: ask_baseline(conn=db_conn, question=q)
    evaluate(question_with_answers, expected_answers, ask_fn)


def evaluate(
    question_with_answers: list[dict],
    expected_answers: dict[str, pd.DataFrame],
    ask_fn,
):
    report = defaultdict[str, defaultdict[str, float]](
        lambda: defaultdict[str, float](float)
    )
    for question in question_with_answers:
        for _ in range(NUM_OF_RUNS):
            response: Response = ask_fn(question["question"])
            report[question["id"]]["total_latency"] += response.latency_s
            report[question["id"]]["total_cost"] += response.cost_usd
            if (
                response.text_to_sql_tool_turn
                and response.text_to_sql_tool_turn.rows is not None
                and not response.text_to_sql_tool_turn.rows.empty
            ):
                actual_rows = response.text_to_sql_tool_turn.rows
                if is_matching(actual_rows, expected_answers[question["id"]]):
                    report[question["id"]]["correct_response"] += 1

    print_report(report, question_with_answers)


def main():
    db_conn: sqlite3.Connection = load_db()
    question_with_answers = json.loads(QUESTION_WITH_ANSWERS_PATH.read_text())
    expected_answers = load_expected_answers(question_with_answers)

    evaluate_agent(db_conn, question_with_answers, expected_answers)
    # evaluate_baseline(db_conn, question_with_answers, expected_answers)


if __name__ == "__main__":
    main()
