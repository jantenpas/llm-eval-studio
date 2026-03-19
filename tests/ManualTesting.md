# Manual API Test Guide

Base URL: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

## Prerequisites

```bash
rm eval_studio.db
uv run fastapi dev api/main.py
```

---

## Step 1 — Create a Project

**`POST /projects`**

```json
{
  "name": "My Assistant v1",
  "description": "Testing my assistant prompt iterations"
}
```

**Save:** `project_id` from the response.

---

## Step 2 — Create a Test Suite

**`POST /projects/{project_id}/suites`**

```json
{
  "name": "Core Q&A"
}
```

**Save:** `suite_id` from the response.

---

## Step 3 — Add Test Cases

**`POST /suites/{suite_id}/test-cases`** — run this 3 times with the payloads below.

**Test Case 1** (exact match — should pass):
```json
{
  "input": "What is the capital of France?",
  "expected_output": "Paris",
  "scoring_method": "exact_match",
  "tags": ["geography"]
}
```

**Test Case 2** (exact match — will likely fail, Claude is verbose):
```json
{
  "input": "What is 2 + 2?",
  "expected_output": "4",
  "scoring_method": "exact_match",
  "tags": ["math"]
}
```

**Test Case 3** (llm_judge — evaluates with reasoning):
```json
{
  "input": "Explain what a REST API is in one sentence.",
  "expected_output": "A REST API is a web service that uses HTTP methods to perform operations on resources identified by URLs.",
  "scoring_method": "llm_judge",
  "tags": ["concepts"]
}
```

No IDs to save here — test cases are read from the suite at run time.

---

## Step 4 — Fire Run 1

**`POST /runs`**

```json
{
  "name": "Prompt v1 — concise",
  "project_id": "<project_id>",
  "suite_id": "<suite_id>",
  "system_prompt": "You are a helpful assistant. Answer as concisely as possible — one word or number when the answer allows.",
  "pass_threshold": 0.70
}
```

**Save:** `run_1_id` from the response.

---

## Step 5 — Poll Until Complete

**`GET /runs/{run_1_id}`** — refresh until `status` is `completed`.

Expected shape when done:
```json
{
  "status": "completed",
  "avg_score": 0.xx,
  "passed": true,
  "results": [ ... ]
}
```

Check each result's `score`, `actual_output`, and `reasoning` (for the llm_judge case).

---

## Step 6 — Check the Quality Gate

**`GET /runs/{run_1_id}/gate`**

Expected:
```json
{
  "passed": true,
  "score": 0.xx,
  "threshold": 0.7,
  "status": "completed"
}
```

---

## Step 7 — Fire Run 2 (Different Prompt)

**`POST /runs`**

```json
{
  "name": "Prompt v2 — verbose",
  "project_id": "<project_id>",
  "suite_id": "<suite_id>",
  "system_prompt": "You are a helpful assistant. Always give thorough, detailed answers.",
  "pass_threshold": 0.70
}
```

**Save:** `run_2_id` from the response. Poll `GET /runs/{run_2_id}` until complete.

---

## Step 8 — Compare the Two Runs

**`GET /runs/{run_1_id}/compare/{run_2_id}`**

Look for `change` values per test case: `improved`, `regressed`, or `unchanged`.
`score_delta` shows the overall shift (positive = run 2 is better).

---

## Step 9 — Verify the Project Overview

**`GET /projects/{project_id}`**

Should show both suites and the 2 recent runs with their scores.

**`GET /projects`**

Should show `latest_run_status`, `latest_run_score`, and `run_count: 2`.

---

## Step 10 — Export the Suite

**`GET /suites/{suite_id}/export`**

Should download `suite-{suite_id}.json` — an array of all 3 test cases, ready to re-import elsewhere.

---

## Error Path Checks

| Endpoint | Expected |
|----------|----------|
| `POST /runs` with `suite_id` pointing to an empty suite | `409` |
| `GET /runs/{run_1_id}/gate` before run completes | `422` |
| `GET /runs/bad-id` | `404` |
| `DELETE /projects/{project_id}` while a run is `running` | `409` |
| `DELETE /suites/{suite_id}` after it has been used in a run | `409` |
| `POST /suites/{suite_id}/test-cases` with `"scoring_method": "invalid"` | `422` |

---

## Test Results — March 19

All tests run against a fresh DB. Server started with `uv run fastapi dev api/main.py`.

---

### Scenario 1 — Exact Match: Concise Prompt

**Result: PASS**

| Input | Expected | Actual | Score |
|-------|----------|--------|-------|
| What is the capital of Japan? | Tokyo | Tokyo | 1.0 |
| What is the capital of Germany? | Berlin | Berlin | 1.0 |
| What is the capital of Australia? | Canberra | Canberra | 1.0 |

- `avg_score`: 1.0
- `passed`: true
- Gate: `passed: true`, score 1.0, threshold 0.8

---

### Scenario 2 — Exact Match: Verbose Prompt

**Result: PASS** (gate correctly rejected the run)

| Input | Expected | Actual (truncated) | Score |
|-------|----------|--------------------|-------|
| What is 5 + 3? | 8 | `## Solving 5 + 3\n**Step 1:** Start with the numbe...` | 0.0 |
| What is 10 - 4? | 6 | `## Solving 10 - 4\n**Step 1:** Start with the numb...` | 0.0 |
| What is 3 × 3? | 9 | `## Solving 3 × 3\nMultiplication is essentially **...` | 0.0 |

- `avg_score`: 0.0
- `passed`: false
- Gate: `passed: false`, score 0.0, threshold 0.8 ✓

---

### Scenario 3 — LLM Judge with Scoring Config Rubric

**Result: PASS**

| Input | Score | Reasoning (truncated) |
|-------|-------|-----------------------|
| Explain what a database index is. | 1.0 | "The response clearly mentions that indexes speed up queries, explicitly discusses trade-offs with write speed and storage, and uses both a book index analogy and a concrete SQL example." |
| What is the difference between SQL and NoSQL? | 1.0 | "The response mentions schema flexibility (rigid vs flexible schema), scalability differences (vertical vs horizontal scaling), and names multiple examples of both SQL (PostgreSQL, MySQL...) and NoSQL types." |

- `avg_score`: 1.0
- `passed`: true (threshold 0.65)
- `scoring_config` rubric was correctly applied — `expected_output: "placeholder"` was ignored

---

### Scenario 4 — Run Comparison: Catch a Regression

**Result: PASS** (regression detected correctly)

- Run A (`Answer with the city name only`): `avg_score` 1.0
- Run B (`You are a helpful assistant`): `avg_score` 0.0
- `score_delta`: -1.0
- All 3 test cases show `"change": "regressed"` with `delta: -1.0`

Full compare response:
```json
{
  "score_delta": -1.0,
  "result_count": 3,
  "results": [
    { "input": "What is the capital of Australia?", "score_a": 1.0, "score_b": 0.0, "change": "regressed" },
    { "input": "What is the capital of Germany?",   "score_a": 1.0, "score_b": 0.0, "change": "regressed" },
    { "input": "What is the capital of Japan?",     "score_a": 1.0, "score_b": 0.0, "change": "regressed" }
  ]
}
```

---

### Scenario 5 — Error Path Gauntlet

| # | Request | Expected | Actual | Pass? |
|---|---------|----------|--------|-------|
| 1 | `GET /runs/does-not-exist` | 404 | 404 | ✓ |
| 2 | `GET /projects/does-not-exist` | 404 | 404 | ✓ |
| 3 | `POST /suites/{id}/test-cases` with `scoring_method: "magic"` | 422 | 422 — "scoring_method must be one of ['exact_match', 'llm_judge']" | ✓ |
| 4 | `POST /runs` with empty suite | 409 | 409 — "Suite has no test cases" | ✓ |
| 5 | `GET /runs/{id}/gate` while run is pending/running | 422 | 422 — "Run is not yet completed" | ✓ |
| 6 | `DELETE /suites/{id}` after used in a run | 409 | 409 — "Suite has been used in runs" | ✓ |
| 7 | `DELETE /projects/{id}` with running run | 409 | 409 — "Project has runs currently in progress" | ✓ |

All 7 error cases returned the correct status codes and messages.

---

### Summary

| Scenario | Result |
|----------|--------|
| 1 — Exact match, concise prompt | ✓ Pass |
| 2 — Exact match, verbose prompt (gate fails) | ✓ Pass |
| 3 — LLM judge with rubric | ✓ Pass |
| 4 — Run comparison / regression detection | ✓ Pass |
| 5 — Error path gauntlet (7/7) | ✓ Pass |

No bugs found. All endpoints behaved as documented.
