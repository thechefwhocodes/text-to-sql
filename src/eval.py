"""Score our agent against the customer's baseline. Run with: python -m src.eval

Runs both arms over the 10 dev questions RUNS times each and averages the
results — accuracy on 10 questions is noisy enough that 8/10 and 9/10 aren't
reliably different from a single run.

GPT-5.4 answers are cached to data/eval_cache.json, keyed by question and run
number, so re-running this script doesn't re-charge for them.
"""

import json
from pathlib import Path

from src.agent import Agent
from src.baseline import ask_baseline
from src.llm import LLM
from src.utils import load_db

QUESTIONS_PATH = Path("data/dev_questions_with_answers.json")
CACHE_PATH = Path("data/eval_cache.json")
ANSWERS_PATH = Path("dev_answers.json")

AGENT_MODEL = "gpt-oss-120b"
BASELINE_MODEL = "gpt-5.4"
RUNS = 3
QUERIES_PER_DAY = 30_000


def is_correct(rows, expected) -> bool:
    """Same values as the gold answer — column names, float noise, and row order
    are all ignored, since many correct queries can shape a result differently."""

    def cell(v):
        is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
        return str(round(v, 2)) if is_number else str(v)

    def bag(result):
        return sorted(tuple(cell(v) for v in row.values()) for row in result)

    return rows is not None and bag(rows) == bag(expected)


def to_record(qid: str, answer) -> dict:
    return {
        "id": qid,
        "sql": answer.sql,
        "text": answer.text,
        "rows": None if answer.rows is None else answer.rows.to_dict("records"),
        "latency_s": answer.latency_s,
        "cost_usd": answer.cost_usd,
    }


def score(records: list[dict], expected: dict) -> dict:
    correct = sum(is_correct(r["rows"], expected[r["id"]]) for r in records)
    latencies = [r["latency_s"] for r in records]
    return {
        "accuracy": correct / len(records),
        "avg_latency_s": sum(latencies) / len(latencies),
        "avg_cost_usd": sum(r["cost_usd"] for r in records) / len(records),
    }


def run_agent_once(conn, questions: list[dict], llm: LLM) -> list[dict]:
    return [
        to_record(q["id"], Agent(conn, model=AGENT_MODEL, llm=llm).ask(q["question"]))
        for q in questions
    ]


def run_baseline_once(conn, questions, run_idx: int, cache: dict, llm: LLM) -> list[dict]:
    """The customer's prompt, reusing a cached answer for this run if we have one."""
    records = []
    for q in questions:
        key = f"{q['id']}_{run_idx}"
        if key not in cache:
            answer = ask_baseline(conn, llm, q["question"], model=BASELINE_MODEL)
            cache[key] = to_record(q["id"], answer)
        records.append(cache[key])
    return records


def report(name: str, runs: list[dict]) -> None:
    print(f"\n{name} — {len(runs)} runs")
    for i, r in enumerate(runs, 1):
        print(
            f"  run {i}: {r['accuracy'] * 10:.0f}/10   "
            f"{r['avg_latency_s']:.2f}s/query   ${r['avg_cost_usd']:.5f}/query"
        )

    avg_acc = sum(r["accuracy"] for r in runs) / len(runs)
    avg_lat = sum(r["avg_latency_s"] for r in runs) / len(runs)
    avg_cost = sum(r["avg_cost_usd"] for r in runs) / len(runs)
    print(
        f"  avg  : {avg_acc * 10:.1f}/10   {avg_lat:.2f}s/query   "
        f"${avg_cost:.5f}/query   ${avg_cost * QUERIES_PER_DAY:.2f}/day at {QUERIES_PER_DAY:,} queries"
    )


def report_failures(name: str, records_per_run: list[list[dict]], expected: dict) -> None:
    """Which questions this arm got wrong most often, across all runs."""
    misses = {}
    for records in records_per_run:
        for r in records:
            if not is_correct(r["rows"], expected[r["id"]]):
                misses[r["id"]] = misses.get(r["id"], 0) + 1

    if misses:
        summary = ", ".join(f"{qid} ({n}/{len(records_per_run)})" for qid, n in sorted(misses.items()))
        print(f"  missed: {summary}")


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    expected = {q["id"]: q["expected_result"] for q in questions}
    conn = load_db()
    llm = LLM()

    agent_records = [run_agent_once(conn, questions, llm) for _ in range(RUNS)]
    report(f"Our agent ({AGENT_MODEL})", [score(r, expected) for r in agent_records])
    report_failures(AGENT_MODEL, agent_records, expected)

    answers = {r["id"]: {"sql": r["sql"], "answer": r["text"]} for r in agent_records[0]}
    ANSWERS_PATH.write_text(json.dumps(answers, indent=2))
    print(f"\nWrote {ANSWERS_PATH}")

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    # Earlier cache entries were keyed by question id alone (a single run). Reuse
    # that as run 0 so this script doesn't re-charge for answers we already have.
    for q in questions:
        if q["id"] in cache and f"{q['id']}_0" not in cache:
            cache[f"{q['id']}_0"] = cache[q["id"]]

    try:
        baseline_records = [
            run_baseline_once(conn, questions, i, cache, llm) for i in range(RUNS)
        ]
    except Exception as e:
        print(f"\nSkipping baseline arm: {e}")
    else:
        CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))
        report(f"Customer baseline ({BASELINE_MODEL})", [score(r, expected) for r in baseline_records])
        report_failures(BASELINE_MODEL, baseline_records, expected)

    conn.close()


if __name__ == "__main__":
    main()
