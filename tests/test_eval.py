import pandas as pd

from src.agent import Response
from src.eval import _from_cache, _to_cache, is_correct, score_run, summarize
from src.turns import TextToSQLToolTurn


def make_answer(rows=None, latency_s=1.0, cost_usd=0.01):
    tool_turn = None
    if rows is not None:
        tool_turn = TextToSQLToolTurn(
            tool_call_id="call_1", content="", sql="SELECT 1", rows=pd.DataFrame(rows)
        )
    return Response(
        text="",
        latency_s=latency_s,
        cost_usd=cost_usd,
        text_to_sql_tool_turn=tool_turn,
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


def test_is_correct_true_when_actual_has_an_extra_unrequested_column():
    # real case: agent's q_005 answer included EmployeeId alongside the
    # requested name and count
    answer = make_answer(
        rows=[
            {
                "EmployeeId": 3,
                "FirstName": "Jane",
                "LastName": "Peacock",
                "CustomerCount": 21,
            }
        ]
    )
    expected = [{"EmployeeName": "Jane Peacock", "CustomerCount": 21}]
    assert is_correct(answer, expected)


def test_is_correct_true_when_gold_column_is_split_across_actual_columns():
    # real case: agent's q_009 answer split CustomerName into FirstName/LastName
    answer = make_answer(
        rows=[
            {"FirstName": "Helena", "LastName": "Holý", "TotalSpent": 49.62, "Rank": 1}
        ]
    )
    expected = [{"CustomerName": "Helena Holý", "TotalSpent": 49.62, "Rank": 1}]
    assert is_correct(answer, expected)


def test_is_correct_false_when_row_count_differs():
    answer = make_answer(rows=[{"a": 1}, {"a": 2}])
    assert not is_correct(answer, [{"a": 1}])


def test_is_correct_false_when_extra_columns_dont_cover_full_expected_value():
    # actual has "Jane" but nowhere has "Peacock" -- must not match
    answer = make_answer(rows=[{"FirstName": "Jane", "CustomerCount": 21}])
    expected = [{"EmployeeName": "Jane Peacock", "CustomerCount": 21}]
    assert not is_correct(answer, expected)


def test_is_correct_matches_each_expected_row_to_a_distinct_actual_row():
    # two identical-looking expected rows must each consume their own actual row
    answer = make_answer(rows=[{"a": 1}, {"a": 1}])
    expected = [{"a": 1}, {"a": 1}]
    assert is_correct(answer, expected)

    answer_missing_one = make_answer(rows=[{"a": 1}, {"a": 2}])
    assert not is_correct(answer_missing_one, expected)


def test_summarize_computes_accuracy_and_misses():
    questions = [
        {"id": "q1", "expected_result": [{"a": 1}]},
        {"id": "q2", "expected_result": [{"a": 1}]},
    ]
    answers = [
        make_answer(rows=[{"a": 1}]),
        make_answer(rows=[{"a": 2}]),
    ]  # q1 right, q2 wrong

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


def test_cache_round_trip_preserves_correctness_for_a_successful_answer():
    answer = make_answer(rows=[{"a": 1}])

    restored = _from_cache(_to_cache(answer))

    assert is_correct(restored, [{"a": 1}])
    assert restored.text_to_sql_tool_turn.sql == "SELECT 1"


def test_cache_round_trip_survives_a_question_that_never_called_the_tool():
    """The baseline sometimes doesn't produce runnable SQL at all — the cache
    must round-trip a None tool turn, not crash on it."""
    answer = make_answer(rows=None)

    restored = _from_cache(_to_cache(answer))

    assert restored.text_to_sql_tool_turn is None
    assert not is_correct(restored, [{"a": 1}])
