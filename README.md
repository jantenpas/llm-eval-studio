# LLM Eval Studio

A lightweight evaluation engine for LLM applications. Write test cases in JSON, run them against Claude, and get scored results with pass/fail reasoning — so you can iterate on prompts with confidence.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An Anthropic API key ([get one here](https://console.anthropic.com))

---

## Setup

```bash
git clone https://github.com/jtenpas/llm-eval-studio
cd llm-eval-studio
uv sync
cp .env.example .env
# Add your Anthropic API key to .env: ANTHROPIC_API_KEY=sk-...
```

---

## Usage

**Run the sample eval:**

```bash
uv run python -m eval_runner.runner
```

**Run the test suite:**

```bash
uv run pytest
```

**Write your own test cases** in `eval_runner/test_cases/` as a JSON array:

```json
[
  {
    "input": "What is the capital of France?",
    "expected_output": "Paris",
    "scoring_method": "exact_match",
    "tags": ["geography"]
  },
  {
    "input": "Explain what a REST API is in one sentence.",
    "expected_output": "A REST API uses HTTP methods to perform operations on resources, returning structured data.",
    "scoring_method": "llm_judge",
    "tags": ["technical"]
  }
]
```

Supported scoring methods:
- `exact_match` — case-insensitive string comparison, strips whitespace
- `llm_judge` — uses Claude to score the response against the expected output (0.0–1.0)

**Run against a custom test suite from Python:**

```python
from pathlib import Path
from eval_runner.runner import run_eval

run_eval(
    test_cases_path=Path("eval_runner/test_cases/my_suite.json"),
    run_name="my-prompt-v2",
    system_prompt="You are a helpful assistant that answers concisely.",
)
```

**Example output:**

```
Starting run: 'sample-run-v1'  (3 test cases)
  [1/3] What is the capital of France?...
  [2/3] Explain what a REST API is in one sentence....
  [3/3] Write a haiku about programming....

=======================================================
  EVAL RESULTS — 2026-02-19 18:22:52 UTC
=======================================================

✗ FAIL  [0.00]  What is the capital of France?...
         Expected 'Paris', got 'The capital of France is **Paris**.'

✓ PASS  [0.80]  Explain what a REST API is in one sentence....
         Captures core REST concepts but omits JSON as the return format.

✓ PASS  [1.00]  Write a haiku about programming....
         Follows 5-7-5 structure and relates clearly to software development.

-------------------------------------------------------
  Passed:      2/3
  Avg Score:   0.60
  Avg Latency: 1319ms
=======================================================
```

---

## Project Structure

```
llm-eval-studio/
  eval_runner/
    models.py        # Pydantic models: Project, TestCase, Run, Result
    runner.py        # Eval engine: load → run → score → report
    test_cases/      # JSON test suite files
    results/         # Timestamped JSON output (gitignored)
  tests/
    test_models.py   # Unit tests for data models
    test_graders.py  # Unit tests for scoring functions
```

---

## Why This Exists

Changing a prompt and thinking it got better is not the same as knowing it got better. This tool treats prompt iteration like software development: define expected behavior, measure it, and catch regressions before they ship.

---

## Roadmap

- [ ] FastAPI backend — REST endpoints for managing projects, runs, and results
- [ ] Next.js UI — test case editor, results dashboard, run comparison view
- [ ] Fuzzy match scoring
- [ ] Multi-model support (compare Claude versions side-by-side)
- [ ] Prompt versioning
