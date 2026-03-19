# LLM Eval Studio

A lightweight evaluation engine for LLM applications. Organise test cases into projects and suites, run them against Claude, and get scored results with pass/fail reasoning — so you can iterate on prompts with confidence and catch regressions before they ship.

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An Anthropic API key ([get one here](https://console.anthropic.com))

---

## Setup

```bash
git clone https://github.com/jantenpas/llm-eval-studio
cd llm-eval-studio
uv sync
cp .env.example .env
# Add your Anthropic API key to .env: ANTHROPIC_API_KEY=sk-...
```

---

## Starting the Server

```bash
uv run fastapi dev api/main.py
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

The database is created automatically on first startup. To start fresh:

```bash
rm eval_studio.db && uv run fastapi dev api/main.py
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Project** | Top-level container. Groups suites and tracks run history. |
| **Test Suite** | A named collection of test cases belonging to a project. |
| **Test Case** | A single input / expected output pair with a scoring method. |
| **Run** | Executes a suite against Claude with a given system prompt. Runs execute in the background and persist results to SQLite. |

---

## REST API

A full walkthrough is in [tests/ManualTesting.md](tests/ManualTesting.md). Quick reference:

### Projects

```bash
POST   /projects                      # create
GET    /projects                      # list (includes latest run score + status)
GET    /projects/{id}                 # detail (includes suites and recent runs)
DELETE /projects/{id}                 # delete (blocked if a run is in progress)
```

### Test Suites

```bash
POST   /projects/{project_id}/suites  # create
GET    /projects/{project_id}/suites  # list
DELETE /suites/{id}                   # delete (blocked if suite has been run)
GET    /suites/{id}/export            # download suite as JSON
POST   /suites/{id}/import            # bulk import test cases from JSON
```

### Test Cases

```bash
POST   /suites/{suite_id}/test-cases  # create
GET    /suites/{suite_id}/test-cases  # list
PUT    /test-cases/{id}               # update
DELETE /test-cases/{id}               # delete
```

### Runs

```bash
POST   /runs                          # create + trigger (executes in background)
GET    /projects/{project_id}/runs    # list runs for a project
GET    /runs/{id}                     # detail + results
DELETE /runs/{id}                     # delete
GET    /runs/{id}/gate                # quality gate — pass/fail against threshold
GET    /runs/{id}/compare/{other_id}  # diff two runs test-case by test-case
```

### Example: fire a run

```bash
# 1. Create a project
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "My Assistant", "description": "Prompt iteration"}'

# 2. Create a suite and add test cases (see RunManualTest.md for full flow)

# 3. Fire a run
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prompt v1",
    "project_id": "<project_id>",
    "suite_id": "<suite_id>",
    "system_prompt": "Answer as concisely as possible.",
    "pass_threshold": 0.80
  }'

# 4. Poll until complete
curl http://localhost:8000/runs/<run_id>

# 5. Check the quality gate
curl http://localhost:8000/runs/<run_id>/gate
```

---

## Scoring Methods

| Method | How it works |
|--------|-------------|
| `exact_match` | Case-insensitive string comparison, strips whitespace. Score is 1.0 or 0.0. |
| `llm_judge` | Claude scores the response 0.0–1.0 against `expected_output`, with reasoning. Supports a `scoring_config` rubric (custom criteria list) instead of `expected_output`. |

### `scoring_config` rubric example

```json
{
  "input": "Explain what a database index is.",
  "expected_output": "placeholder",
  "scoring_method": "llm_judge",
  "scoring_config": {
    "criteria": [
      "Mentions that an index speeds up queries",
      "Mentions the trade-off with write speed or storage",
      "Uses an analogy or concrete example"
    ]
  }
}
```

---

## Testing

### Unit tests (no server required)

```bash
uv run pytest
uv run pytest --cov --cov-report=term-missing
```

### E2E tests (requires live server + `ANTHROPIC_API_KEY`)

```bash
uv run fastapi dev api/main.py   # in a separate terminal
uv run pytest tests/e2e/ -v
```

See [tests/e2e/README.md](tests/e2e/README.md) for details.

---

## Project Structure

```
llm-eval-studio/
  api/
    main.py            # FastAPI app and lifespan
    routes.py          # All route handlers
    database.py        # SQLite connection, migrations, query functions
    schemas.py         # Pydantic request/response models
  eval_runner/
    models.py          # Core data models
    runner.py          # Eval engine: call Claude → score → persist
    scorers.py         # Scorer protocol — ExactMatchScorer, LLMJudgeScorer
  migrations/
    001_initial_schema.sql   # DB schema (applied automatically on startup)
  tests/
    unit/              # Fast tests, no server required (TestClient + in-memory DB)
      test_api.py
      test_graders.py
      test_models.py
      test_runner.py
    e2e/               # End-to-end tests against a live server
      test_exact_match.py
      test_llm_judge.py
      test_run_comparison.py
      test_error_paths.py
```

---

## Why This Exists

Changing a prompt and thinking it got better is not the same as knowing it got better. This tool treats prompt iteration like software development: define expected behavior, measure it, and catch regressions before they ship.

---

## Roadmap

- [x] Core eval engine — load, run, score, report
- [x] FastAPI backend — projects, suites, test cases, runs, quality gate, run comparison
- [x] SQLite persistence with migration system
- [x] `llm_judge` scoring with custom rubric support
- [x] E2E regression test suite
- [ ] Next.js UI — test case editor, results dashboard, run comparison view
- [ ] Fuzzy match scoring
- [ ] Multi-model support (compare Claude versions side-by-side)
- [ ] Prompt versioning
