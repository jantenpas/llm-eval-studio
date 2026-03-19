import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

CallClaudeFn = Callable[[str], tuple[str, int]]


@dataclass
class ScoredResult:
    score: float
    passed: bool
    reasoning: str | None


class Scorer(Protocol):
    async def score(
        self,
        input: str,
        expected: str,
        actual: str,
        pass_threshold: float,
        scoring_config: str | None = None,
    ) -> ScoredResult: ...


class ExactMatchScorer:
    async def score(
        self,
        input: str,
        expected: str,
        actual: str,
        pass_threshold: float,
        scoring_config: str | None = None,
    ) -> ScoredResult:
        matched = actual.strip().lower() == expected.strip().lower()
        s = 1.0 if matched else 0.0
        reasoning = "Exact match." if matched else f"Expected '{expected}', got '{actual}'."
        return ScoredResult(score=s, passed=s >= pass_threshold, reasoning=reasoning)


class LLMJudgeScorer:
    """Calls Claude to score actual output against expected output or a rubric."""

    def __init__(self, call_claude_fn: CallClaudeFn) -> None:
        self._call_claude = call_claude_fn

    async def score(
        self,
        input: str,
        expected: str,
        actual: str,
        pass_threshold: float,
        scoring_config: str | None = None,
    ) -> ScoredResult:
        if scoring_config:
            config = json.loads(scoring_config)
            criteria_lines = "\n".join(f"- {c}" for c in config.get("criteria", []))
            prompt = f"""You are evaluating an AI assistant's response against a rubric.

<actual>{actual}</actual>

Evaluate the response against ALL of the following criteria:
{criteria_lines}

Score how well the response satisfies all criteria:
- 1.0 = all criteria met
- 0.5 = some criteria met
- 0.0 = criteria not met

Respond in this exact format:
<reasoning>one sentence explanation</reasoning>
<score>decimal between 0.0 and 1.0</score>"""
        else:
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

        response, _ = self._call_claude(prompt)

        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        score_match = re.search(r"<score>(.*?)</score>", response, re.DOTALL)

        reasoning = (
            reasoning_match.group(1).strip() if reasoning_match else "Could not parse reasoning."
        )
        try:
            s = float(score_match.group(1).strip()) if score_match else 0.0
            s = max(0.0, min(1.0, s))
        except ValueError:
            s = 0.0

        return ScoredResult(score=s, passed=s >= pass_threshold, reasoning=reasoning)


SCORER_REGISTRY: dict[str, type] = {
    "exact_match": ExactMatchScorer,
    "llm_judge": LLMJudgeScorer,
}


def get_scorer(method: str, call_claude_fn: CallClaudeFn | None = None) -> Scorer:
    if method not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scoring method: {method}")
    if method == "llm_judge":
        if call_claude_fn is None:
            raise ValueError("call_claude_fn is required for llm_judge scorer")
        return LLMJudgeScorer(call_claude_fn)
    return ExactMatchScorer()
