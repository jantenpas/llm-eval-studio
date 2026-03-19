import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

from eval_runner.models import Result, Run, RunStatus, TestCase
from eval_runner.scorers import get_scorer

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


async def run_test_case_async(
    test_case: TestCase,
    run: Run,
    pass_threshold: float = PASS_THRESHOLD,
    scoring_config: str | None = None,
) -> Result:
    actual_output, latency_ms = await asyncio.to_thread(
        call_claude, test_case.input, run.system_prompt
    )

    scorer = get_scorer(test_case.scoring_method, call_claude_fn=call_claude)
    scored = await scorer.score(
        input=test_case.input,
        expected=test_case.expected_output,
        actual=actual_output,
        pass_threshold=pass_threshold,
        scoring_config=scoring_config,
    )

    return Result(
        run_id=run.id,
        test_case_id=test_case.id,
        actual_output=actual_output,
        score=scored.score,
        reasoning=scored.reasoning or "",
        latency_ms=latency_ms,
    )


def run_test_case(test_case: TestCase, run: Run) -> Result:
    """Synchronous wrapper used by the legacy run_eval path."""
    actual_output, latency_ms = call_claude(test_case.input, run.system_prompt)

    scorer = get_scorer(test_case.scoring_method, call_claude_fn=call_claude)
    scored = asyncio.run(
        scorer.score(
            input=test_case.input,
            expected=test_case.expected_output,
            actual=actual_output,
            pass_threshold=PASS_THRESHOLD,
        )
    )

    return Result(
        run_id=run.id,
        test_case_id=test_case.id,
        actual_output=actual_output,
        score=scored.score,
        reasoning=scored.reasoning or "",
        latency_ms=latency_ms,
    )


def print_summary(results: list[Result], test_cases: list[TestCase]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.score >= PASS_THRESHOLD)
    avg_score = sum(r.score for r in results) / total if total else 0
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0

    tc_map = {tc.id: tc for tc in test_cases}

    print("\n" + "=" * 55)
    print(f"  EVAL RESULTS — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
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
