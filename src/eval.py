"""
Evaluate our agent against the customer's baseline over the dev question set.
Run with: python -m src.eval
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.agent import Agent, Response
from src.baseline import ask_baseline
from src.llm import LLM
from src.models import GPT_5_4, GPT_OSS_120B
from src.utils import load_db

QUESTIONS_PATH = Path("data/dev_questions_with_answers.json")
CACHE_PATH = Path("data/eval_cache.json")
ANSWERS_PATH = Path("data/dev_answers.json")
NOTES_PATH = Path("NOTES.md")

RUNS = 3
QUERIES_PER_DAY = 30_000

RESULTS_START = "<!-- eval-results:start -->"
RESULTS_END = "<!-- eval-results:end -->"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def is_correct(answer: Response, expected: list[dict]) -> bool:
    """Same values as the gold answer — column names, column count, float
    noise, and row order are all ignored, since many correct queries shape a
    result differently."""
    if answer.rows is None:
        return False

    actual = answer.rows.to_dict("records")
    if len(actual) != len(expected):
        return False

    def cell(v) -> str:
        is_number = isinstance(v, (int, float)) and not isinstance(v, bool)
        return str(round(v, 2)) if is_number else str(v)

    def words(row: dict) -> set[str]:
        return {word for v in row.values() for word in cell(v).split()}

    remaining = list(actual)
    for expected_row in expected:
        needed = words(expected_row)
        match = next((row for row in remaining if needed <= words(row)), None)
        if match is None:
            return False
        remaining.remove(match)

    return True


@dataclass
class Score:
    """One question's result within one run."""

    question_id: str
    correct: bool
    latency_s: float
    cost_usd: float


def score_run(questions: list[dict], answers: list[Response]) -> list[Score]:
    return [
        Score(
            question_id=q["id"],
            correct=is_correct(a, q["expected_result"]),
            latency_s=a.latency_s,
            cost_usd=a.cost_usd,
        )
        for q, a in zip(questions, answers)
    ]


# ---------------------------------------------------------------------------
# Running the two arms
# ---------------------------------------------------------------------------


def run_agent(conn, questions: list[dict], llm: LLM) -> list[Response]:
    """Our agent, one fresh conversation per question."""
    return [
        Agent(conn, model=GPT_OSS_120B, llm=llm).ask(q["question"]) for q in questions
    ]


def run_baseline_cached(
    conn, questions: list[dict], run_idx: int, cache: dict, llm: LLM
) -> list[Response]:
    """The customer's prompt, reusing a cached answer for this run if we have one."""
    answers = []
    for q in questions:
        key = f"{q['id']}_{run_idx}"
        if key not in cache:
            cache[key] = _to_cache(
                ask_baseline(conn, llm, q["question"], model=GPT_5_4)
            )
        answers.append(_from_cache(cache[key]))
    return answers


def _to_cache(answer: Response) -> dict:
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


def _from_cache(entry: dict) -> Response:
    rows = None if entry["rows"] is None else pd.DataFrame(entry["rows"])
    return Response(
        text=entry["text"],
        sql=entry["sql"],
        rows=rows,
        sql_attempts=entry["sql_attempts"],
        latency_s=entry["latency_s"],
        cost_usd=entry["cost_usd"],
    )


# ---------------------------------------------------------------------------
# Aggregating results across runs
# ---------------------------------------------------------------------------


@dataclass
class Summary:
    """Metrics for one arm (agent or baseline), aggregated across all runs."""

    name: str
    num_questions: int
    num_runs: int
    accuracy: float
    avg_latency_s: float
    avg_cost_usd: float
    misses: dict[str, int]  # question id -> number of runs it was wrong


def summarize(name: str, scores: list[Score]) -> Summary:
    """Aggregate every (question, run) score into one Summary."""
    question_ids = {s.question_id for s in scores}

    misses: dict[str, int] = {}
    for s in scores:
        if not s.correct:
            misses[s.question_id] = misses.get(s.question_id, 0) + 1

    return Summary(
        name=name,
        num_questions=len(question_ids),
        num_runs=len(scores) // len(question_ids),
        accuracy=sum(s.correct for s in scores) / len(scores),
        avg_latency_s=sum(s.latency_s for s in scores) / len(scores),
        avg_cost_usd=sum(s.cost_usd for s in scores) / len(scores),
        misses=misses,
    )


def print_summary(summary: Summary) -> None:
    print(
        f"\n{summary.name} — {summary.num_runs} runs x {summary.num_questions} questions"
    )
    print(
        f"  accuracy: {summary.accuracy:.1%}\n"
        f"  latency : {summary.avg_latency_s:.2f}s/query\n"
        f"  cost    : ${summary.avg_cost_usd:.5f}/query  "
        f"-> ${summary.avg_cost_usd * QUERIES_PER_DAY:.2f}/day at {QUERIES_PER_DAY:,} queries"
    )
    if summary.misses:
        detail = ", ".join(
            f"{qid} ({n}/{summary.num_runs})"
            for qid, n in sorted(summary.misses.items())
        )
        print(f"  missed  : {detail}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(summaries: list[Summary], questions: list[dict]) -> str:
    question_text = {q["id"]: q["question"] for q in questions}

    lines = [
        RESULTS_START,
        "## Evaluation Results",
        "",
        "_Auto-generated by `python -m src.eval` — do not edit by hand._",
        "",
        f"{summaries[0].num_runs} runs over {summaries[0].num_questions} dev questions.",
        "",
        "| Arm | Accuracy | Avg Latency/query | Avg Cost/query | Est. $/day @ 30k queries |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.name} | {s.accuracy:.1%} | {s.avg_latency_s:.2f}s | "
            f"${s.avg_cost_usd:.5f} | ${s.avg_cost_usd * QUERIES_PER_DAY:,.2f} |"
        )

    lines += ["", "### Missed Questions", ""]
    for s in summaries:
        lines.append(f"**{s.name}**")
        lines.append("")
        if not s.misses:
            lines.append("_None — every question was answered correctly in every run._")
        else:
            for qid, n in sorted(s.misses.items()):
                lines.append(
                    f"- `{qid}` — wrong {n}/{s.num_runs} runs: {question_text[qid]}"
                )
        lines.append("")

    lines.append(RESULTS_END)
    return "\n".join(lines)


def write_results_section(markdown: str) -> None:
    """Replace the eval-results section in NOTES.md (between markers), or
    append it as a new section if this is the first time the eval has run."""
    text = NOTES_PATH.read_text() if NOTES_PATH.exists() else ""

    if RESULTS_START in text and RESULTS_END in text:
        before = text.split(RESULTS_START)[0]
        after = text.split(RESULTS_END)[1]
        text = before + markdown + after
    else:
        separator = "" if not text or text.endswith("\n\n") else "\n\n"
        text = text + separator + markdown + "\n"

    NOTES_PATH.write_text(text)


def write_dev_answers(questions: list[dict], answers: list[Response]) -> None:
    data = {
        q["id"]: {"sql": a.sql, "answer": a.text} for q, a in zip(questions, answers)
    }
    ANSWERS_PATH.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {ANSWERS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text())
    conn = load_db()
    llm = LLM()

    print(f"Our agent ({GPT_OSS_120B}): {RUNS} runs x {len(questions)} questions")
    agent_runs = [run_agent(conn, questions, llm) for _ in range(RUNS)]
    agent_summary = summarize(
        f"Our agent ({GPT_OSS_120B})",
        [score for run in agent_runs for score in score_run(questions, run)],
    )
    write_dev_answers(questions, agent_runs[0])
    print_summary(agent_summary)

    summaries = [agent_summary]

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    baseline_summary = None
    try:
        print(
            f"\nBaseline ({GPT_5_4}): {RUNS} runs x {len(questions)} questions (cached where possible)"
        )
        baseline_runs = [
            run_baseline_cached(conn, questions, i, cache, llm) for i in range(RUNS)
        ]
    except Exception as e:
        print(f"  skipping baseline arm: {e}")
    else:
        baseline_summary = summarize(
            f"Baseline ({GPT_5_4})",
            [score for run in baseline_runs for score in score_run(questions, run)],
        )
        print_summary(baseline_summary)
    finally:
        # Persist whatever got cached, even if a later question failed.
        CACHE_PATH.write_text(json.dumps(cache, indent=2))

    if baseline_summary:
        summaries.append(baseline_summary)

    conn.close()

    write_results_section(render_markdown(summaries, questions))
    print(f"\nWrote results to {NOTES_PATH}")


if __name__ == "__main__":
    main()
