# Text-to-SQL Agent

Interactive CLI that turns plain-English questions into SQL, runs them against a database, and explains the results back to you. See [NOTES.md](NOTES.md) for what was built, how it was validated, and known gaps.

## Setup

```bash
./setup.sh   # downloads the sample database
uv sync      # installs dependencies
```

## Environment Variables


| Variable            | Required for                                                      |
| ------------------- | ----------------------------------------------------------------- |
| `FIREWORKS_API_KEY` | The CLI, and the agent half of `python -m src.eval`               |
| `OPENAI_API_KEY`    | Only the baseline half of `python -m src.eval` — skipped if unset |


## Run

```bash
export FIREWORKS_API_KEY=<your-key>
uv run cli
```

Type a question, see the SQL and results, get a plain-English answer. Ask a follow-up and it remembers the conversation. Type `exit` or `quit` to leave.

## Evaluate

```bash
uv run python -m src.eval
```

Scores our agent against the customer's baseline on the 10 dev questions and writes `data/dev_answers.json`.

## Test

```bash
uv run pytest
```



## Email to Raul

Subject: Text-to-SQL PoC

Hi Raul,

Quick update on the agentic BI CLI proof of concept — we have it's working, and the numbers back it up.

**What I built**

An interactive CLI that takes a plain-English question, writes SQL against your database, runs it, and explains the result in plain English — including follow-ups in the same session. It runs on `gpt-oss-120b`, an open-source model on Fireworks.

The core fix versus your prototype: instead of asking the model to guess table and column names blind, I give it your real schema up front. And instead of trusting whatever SQL it writes on the first try, the model runs its query as a tool call — if it fails, the error goes back to the model and it gets a few more tries before giving up. That directly targets the "hallucinated table names, invalid SQL" failure mode you flagged.

**How it performed on the 10 dev questions**

I ran your original prompt and this agent 3 times each over all 10 dev questions — one pass isn't reliable enough to trust, since the same question can pass once and fail the next run.


|                          | Accuracy | Avg latency | Cost/query | Est. $/day at 30k queries |
| ------------------------ | -------- | ----------- | ---------- | ------------------------- |
| Your prototype (GPT-5.4) | 46.7%    | 1.34s       | $0.00103   | $30.90                    |
| This PoC (gpt-oss-120b)  | 100.0%   | 1.60s       | $0.00063   | $18.97                    |


Your prototype missed the same 5 of 10 questions in every run — almost entirely wrong column or table guesses, exactly what you described — plus a 6th question it got wrong occasionally. This agent got all 10 right across all 3 runs, comes in under your 3-second latency target, and costs about 39% less per query at your projected scale.

Happy to walk through the code and the eval live, and get into what it takes to harden this for production.

Best,
Aashish