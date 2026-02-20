import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

from eval_runner.models import Result, Run, RunStatus, ScoringMethod, TestCase

MODEL = "claude-sonnet-4-6"
PASS_THRESHOLD = 0.7
RESULTS_DIR = Path(__file__).parent / "results"
TEST_CASES_DIR = Path(__file__).parent / "test_cases"

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic()
    return _client


def load_test_cases(path: Path, project_id: UUID) -> list[TestCase]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array, got {type(raw).__name__}")
    return [TestCase(project_id=project_id, **tc) for tc in raw]


def call_claude(prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> tuple[str, int]:
    start = time.monotonic()
    client = get_client()

    if system_prompt:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
        )
    else:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

    latency_ms = int((time.monotonic() - start) * 1000)

    if not response.content:
        raise ValueError("API returned empty content list")
    block = response.content[0]
    if not isinstance(block, TextBlock):
        raise ValueError(f"Unexpected content block type: {type(block)}")
    return block.text, latency_ms


def grade_exact_match(actual: str, expected: str) -> tuple[float, str]:
    match = actual.strip().lower() == expected.strip().lower()
    score = 1.0 if match else 0.0
    reasoning = "Exact match." if match else f"Expected '{expected}', got '{actual}'."
    return score, reasoning


def grade_with_llm(actual: str, expected: str) -> tuple[float, str]:
    prompt = f"""You are evaluating an AI assistant's response against an expected output.

<expected>{expected}</expected>
<actual>{actual}</actual>

Score how well the actual response satisfies the intent of the expected output.
- 1.0 = correct and complete
- 0.5 = partially correct
- 0.0 = incorrect or irrelevant

Respond in this exact format:
<reasoning>one sentence explanation</reasoning>
<score>decimal between 0.0 and 1.0</score>"""

    response, _ = call_claude(prompt)

    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    score_match = re.search(r"<score>(.*?)</score>", response, re.DOTALL)

    reasoning = (
        reasoning_match.group(1).strip()
        if reasoning_match
        else "Could not parse reasoning."
    )

    try:
        score = float(score_match.group(1).strip()) if score_match else 0.0
        score = max(0.0, min(1.0, score))  # clamp to valid range just in case
    except ValueError:
        score = 0.0

    return score, reasoning


def run_test_case(test_case: TestCase, run: Run) -> Result:
    actual_output, latency_ms = call_claude(test_case.input, run.system_prompt)

    if test_case.scoring_method == ScoringMethod.exact_match:
        score, reasoning = grade_exact_match(actual_output, test_case.expected_output)
    elif test_case.scoring_method == ScoringMethod.llm_judge:
        score, reasoning = grade_with_llm(actual_output, test_case.expected_output)
    else:
        raise NotImplementedError(  # not yet implemented
            f"Scoring method '{test_case.scoring_method}' is not yet implemented."
        )

    return Result(
        run_id=run.id,
        test_case_id=test_case.id,
        actual_output=actual_output,
        score=score,
        reasoning=reasoning,
        latency_ms=latency_ms,
    )


def print_summary(results: list[Result], test_cases: list[TestCase]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.score >= PASS_THRESHOLD)
    avg_score = sum(r.score for r in results) / total if total else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0

    tc_map = {tc.id: tc for tc in test_cases}

    print("\n" + "=" * 55)
    print(
        f"  EVAL RESULTS — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    print("=" * 55)

    for result in results:
        tc = tc_map.get(result.test_case_id)
        status = "✓ PASS" if result.score >= PASS_THRESHOLD else "✗ FAIL"
        input_preview = tc.input[:55] if tc else "unknown"
        print(f"\n{status}  [{result.score:.2f}]  {input_preview}...")
        print(f"         {result.reasoning}")

    print("\n" + "-" * 55)
    print(f"  Passed:      {passed}/{total}")
    print(f"  Avg Score:   {avg_score:.2f}")
    print(f"  Avg Latency: {avg_latency:.0f}ms")
    print("=" * 55 + "\n")


def save_results(results: list[Result], run: Run) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    filename = f"{run.name.replace(' ', '_')}_{run.id}.json"
    output_path = RESULTS_DIR / filename

    output = {
        "run": run.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved → {output_path}\n")


def run_eval(test_cases_path: Path, run_name: str, system_prompt: str = "") -> list[Result]:
    project_id = uuid4()
    test_cases = load_test_cases(test_cases_path, project_id)

    run = Run(
        project_id=project_id,
        name=run_name,
        llm_model=MODEL,
        system_prompt=system_prompt,
        status=RunStatus.running,
    )

    print(f"\nStarting run: '{run.name}'  ({len(test_cases)} test cases)")

    results: list[Result] = []
    errors: list[str] = []
    for i, tc in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] {tc.input[:55]}...")
        try:
            result = run_test_case(tc, run)
            results.append(result)
        except Exception as e:
            errors.append(str(e))
            print(f"  ERROR on test case {i} ({type(e).__name__}): {e}")

    run.status = RunStatus.failed if errors else RunStatus.completed
    print_summary(results, test_cases)
    save_results(results, run)
    return results


if __name__ == "__main__":  # pragma: no cover
    run_eval(
        test_cases_path=TEST_CASES_DIR / "sample.json",
        run_name="sample-run-v1",
    )
