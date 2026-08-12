# Text-to-SQL Agent

Interactive CLI that turns plain-English questions into SQL, runs them against a database, and explains the results back to you. See [NOTES.md](NOTES.md) for what was built, how it was validated, and known gaps.

## Setup

```bash
./setup.sh   # downloads the sample database
uv sync      # installs dependencies
```

## Environment Variables

| Variable | Required for |
|---|---|
| `FIREWORKS_API_KEY` | The CLI, and the agent half of `python -m src.eval` |
| `OPENAI_API_KEY` | Only the baseline half of `python -m src.eval` — skipped if unset |

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
