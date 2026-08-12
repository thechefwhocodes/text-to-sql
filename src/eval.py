"""Score our agent against the customer's baseline. Run with: python -m src.eval

Runs both arms over the 10 dev questions RUNS times each and averages the
results — accuracy on 10 questions is noisy enough that a single run isn't
reliable. GPT-5.4 answers are cached to data/eval_cache.json, so re-running
this script doesn't re-charge for them.
"""

import json
from pathlib import Path

import pandas as pd

from src.agent import Agent, Answer
from src.baseline import ask_baseline
from src.llm import LLM
from src.models import GPT_5_4, GPT_OSS_120B
from src.utils import load_db

QUESTIONS_PATH = Path("data/dev_questions_with_answers.json")
CACHE_PATH = Path("data/eval_cache.json")
ANSWERS_PATH = Path("data/dev_answers.json")

RUNS = 3
QUERIES_PER_DAY = 30_000


def is_correct(answer: Answer, expected: list[dict]) -> bool:
    """Same values as the gold answer — column names, float noise, and row order
    are all ignored, since many correct queries shape a result differently."""
    if answer.rows is None:
        return False

    def cell(v):
        is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
        return str(round(v, 2)) if is_number else str(v)

    def bag(records):
        return sorted(tuple(cell(v) for v in row.values()) for row in records)

    return bag(answer.rows.to_dict("records")) == bag(expected)


def run_agent(conn, questions: list[dict], llm: LLM) -> list[Answer]:
    """Our agent, one fresh conversation per question."""
    return [
        Agent(conn, model=GPT_OSS_120B, llm=llm).ask(q["question"]) for q in questions
    ]


def run_baseline(conn, questions: list[dict], run_idx: int, cache: dict, llm: LLM) -> list[Answer]:
    """The customer's prompt, reusing a cached answer for this run if we have one."""
    answers = []
    for q in questions:
        key = f"{q['id']}_{run_idx}"
        if key not in cache:
            cache[key] = to_cache(ask_baseline(conn, llm, q["question"], model=GPT_5_4))
        answers.append(from_cache(cache[key]))
    return answers


def to_cache(answer: Answer) -> dict:
    """Flatten an Answer to plain JSON — `rows` is a DataFrame, which isn't
    serialisable, and everything else is already a plain type."""
    return {
        "text": answer.text,
        "sql": answer.sql,
        "rows": None if answer.rows is None else answer.rows.to_dict("records"),
        "sql_attempts": answer.sql_attempts,
        "latency_s": answer.latency_s,
        "cost_usd": answer.cost_usd,
    }


def from_cache(entry: dict) -> Answer:
    rows = None if entry["rows"] is None else pd.DataFrame(entry["rows"])
    return Answer(
        text=entry["text"],
        sql=entry["sql"],
        rows=rows,
        sql_attempts=entry["sql_attempts"],
        latency_s=entry["latency_s"],
        cost_usd=entry["cost_usd"],
    )


def summarize(questions: list[dict], answers: list[Answer]) -> dict:
    """Accuracy, latency and cost for one run over all 10 questions."""
    correct = sum(is_correct(a, q["expected_result"]) for q, a in zip(questions, answers))
    return {
        "correct": correct,
        "avg_latency_s": sum(a.latency_s for a in answers) / len(answers),
        "avg_cost_usd": sum(a.cost_usd for a in answers) / len(answers),
    }


def report(name: str, questions: list[dict], runs: list[list[Answer]]) -> None:
    """Print per-run and averaged results, plus which questions failed and how often."""
    print(f"\n{name} — {len(runs)} runs")

    summaries = [summarize(questions, answers) for answers in runs]
    for i, s in enumerate(summaries, 1):
        print(
            f"  run {i}: {s['correct']}/{len(questions)}   "
            f"{s['avg_latency_s']:.2f}s/query   ${s['avg_cost_usd']:.5f}/query"
        )

    avg_correct = sum(s["correct"] for s in summaries) / len(summaries)
    avg_latency = sum(s["avg_latency_s"] for s in summaries) / len(summaries)
    avg_cost = sum(s["avg_cost_usd"] for s in summaries) / len(summaries)
    print(
        f"  avg  : {avg_correct:.1f}/{len(questions)}   {avg_latency:.2f}s/query   "
        f"${avg_cost:.5f}/query   ${avg_cost * QUERIES_PER_DAY:.2f}/day at {QUERIES_PER_DAY:,} queries"
    )

    misses = {}
    for answers in runs:
        for q, a in zip(questions, answers):
            if not is_correct(a, q["expected_result"]):
                misses[q["id"]] = misses.get(q["id"], 0) + 1
    if misses:
        summary = ", ".join(f"{qid} ({n}/{len(runs)})" for qid, n in sorted(misses.items()))
        print(f"  missed: {summary}")


def write_dev_answers(questions: list[dict], answers: list[Answer]) -> None:
    data = {q["id"]: {"sql": a.sql, "answer": a.text} for q, a in zip(questions, answers)}
    ANSWERS_PATH.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {ANSWERS_PATH}")


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    conn = load_db()
    llm = LLM()

    agent_runs = [run_agent(conn, questions, llm) for _ in range(RUNS)]
    report(f"Our agent ({GPT_OSS_120B})", questions, agent_runs)
    write_dev_answers(questions, agent_runs[0])

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    try:
        baseline_runs = [run_baseline(conn, questions, i, cache, llm) for i in range(RUNS)]
    except Exception as e:
        print(f"\nSkipping baseline arm: {e}")
    else:
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
        report(f"Customer baseline ({GPT_5_4})", questions, baseline_runs)

    conn.close()


if __name__ == "__main__":
    main()
