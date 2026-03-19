# E2E Tests

End-to-end tests that run against a live server with real LLM calls.

## Requirements

- `ANTHROPIC_API_KEY` set in your environment
- Server running locally: `uv run fastapi dev api/main.py`
- Fresh DB recommended: `rm eval_studio.db` before running

## Running

```bash
# E2E only
uv run pytest tests/e2e/ -v

# Everything (unit + e2e)
uv run pytest tests/ -v

# Unit tests only (default — no server needed)
uv run pytest
```

## Notes

- Tests create their own projects, suites, and runs — no shared fixtures between files
- Data is not cleaned up after each run; wipe the DB between full suite runs to avoid accumulation
- LLM judge tests are slower (~30s per run) due to real Claude calls
- `test_gate_on_incomplete_run_returns_422` is timing-sensitive; it hits the gate immediately after firing a run and relies on the run not completing instantly
