import pandas as pd

from src.agent import Answer
from src.eval import is_correct, score_run, summarize


def make_answer(rows=None, latency_s=1.0, cost_usd=0.01):
    return Answer(
        text="",
        sql=None,
        rows=None if rows is None else pd.DataFrame(rows),
        sql_attempts=1,
        latency_s=latency_s,
        cost_usd=cost_usd,
    )


def test_is_correct_ignores_row_order():
    answer = make_answer(rows=[{"a": 2}, {"a": 1}])
    expected = [{"a": 1}, {"a": 2}]
    assert is_correct(answer, expected)


def test_is_correct_ignores_float_noise():
    answer = make_answer(rows=[{"total": 1.2300000001}])
    expected = [{"total": 1.23}]
    assert is_correct(answer, expected)


def test_is_correct_ignores_column_names():
    answer = make_answer(rows=[{"different_name": 1}])
    expected = [{"a": 1}]
    assert is_correct(answer, expected)


def test_is_correct_false_when_values_differ():
    answer = make_answer(rows=[{"a": 1}])
    assert not is_correct(answer, [{"a": 2}])


def test_is_correct_false_when_no_rows():
    answer = make_answer(rows=None)
    assert not is_correct(answer, [{"a": 1}])


def test_summarize_computes_accuracy_and_misses():
    questions = [
        {"id": "q1", "expected_result": [{"a": 1}]},
        {"id": "q2", "expected_result": [{"a": 1}]},
    ]
    answers = [make_answer(rows=[{"a": 1}]), make_answer(rows=[{"a": 2}])]  # q1 right, q2 wrong

    summary = summarize("test", score_run(questions, answers))

    assert summary.accuracy == 0.5
    assert summary.num_runs == 1
    assert summary.num_questions == 2
    assert summary.misses == {"q2": 1}


def test_summarize_counts_misses_per_question_across_runs():
    questions = [{"id": "q1", "expected_result": [{"a": 1}]}]
    run1 = score_run(questions, [make_answer(rows=[{"a": 1}])])  # correct
    run2 = score_run(questions, [make_answer(rows=[{"a": 2}])])  # wrong
    run3 = score_run(questions, [make_answer(rows=[{"a": 2}])])  # wrong

    summary = summarize("test", run1 + run2 + run3)

    assert summary.num_runs == 3
    assert summary.misses == {"q1": 2}


def test_summarize_with_no_misses_reports_empty_dict():
    questions = [{"id": "q1", "expected_result": [{"a": 1}]}]
    scores = score_run(questions, [make_answer(rows=[{"a": 1}])])

    summary = summarize("test", scores)

    assert summary.accuracy == 1.0
    assert summary.misses == {}
