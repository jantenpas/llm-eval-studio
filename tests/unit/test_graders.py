"""Tests for scorer implementations (previously test_graders.py)."""
import pytest

from eval_runner.scorers import ExactMatchScorer, LLMJudgeScorer


class TestExactMatchScorer:
    scorer = ExactMatchScorer()

    async def test_exact_match_passes(self) -> None:
        r = await self.scorer.score("Q", "Paris", "Paris", pass_threshold=0.7)
        assert r.score == 1.0
        assert r.passed is True

    async def test_exact_match_case_insensitive(self) -> None:
        r = await self.scorer.score("Q", "Paris", "paris", pass_threshold=0.7)
        assert r.score == 1.0

    async def test_exact_match_fails_on_mismatch(self) -> None:
        r = await self.scorer.score("Q", "Paris", "London", pass_threshold=0.7)
        assert r.score == 0.0
        assert r.passed is False

    async def test_exact_match_strips_whitespace(self) -> None:
        r = await self.scorer.score("Q", "Paris", "  Paris  ", pass_threshold=0.7)
        assert r.score == 1.0

    async def test_fails_on_verbose_response(self) -> None:
        r = await self.scorer.score(
            "Q", "Paris", "The capital of France is Paris.", pass_threshold=0.7
        )
        assert r.score == 0.0

    async def test_reasoning_on_pass(self) -> None:
        r = await self.scorer.score("Q", "Paris", "Paris", pass_threshold=0.7)
        assert r.reasoning == "Exact match."

    async def test_reasoning_on_fail_includes_expected_and_actual(self) -> None:
        r = await self.scorer.score("Q", "Paris", "London", pass_threshold=0.7)
        assert r.reasoning is not None
        assert "Paris" in r.reasoning
        assert "London" in r.reasoning

    async def test_passed_respects_threshold(self) -> None:
        # score=1.0, threshold=0.7 → passed
        r = await self.scorer.score("Q", "A", "A", pass_threshold=0.7)
        assert r.passed is True
        # score=0.0, threshold=0.7 → not passed
        r2 = await self.scorer.score("Q", "A", "B", pass_threshold=0.7)
        assert r2.passed is False


class TestLLMJudgeScorer:
    def _mock_call(self, reasoning: str, score: str):  # type: ignore[no-untyped-def]
        def call_claude(prompt: str) -> tuple[str, int]:
            return f"<reasoning>{reasoning}</reasoning>\n<score>{score}</score>", 0
        return call_claude

    async def test_parses_score_and_reasoning(self) -> None:
        scorer = LLMJudgeScorer(self._mock_call("Looks correct.", "0.9"))
        r = await scorer.score("Q", "expected", "actual", pass_threshold=0.7)
        assert r.score == pytest.approx(0.9)
        assert r.reasoning == "Looks correct."

    async def test_clamps_score_above_one(self) -> None:
        scorer = LLMJudgeScorer(self._mock_call("Perfect.", "1.5"))
        r = await scorer.score("Q", "a", "b", pass_threshold=0.7)
        assert r.score == 1.0

    async def test_clamps_score_below_zero(self) -> None:
        scorer = LLMJudgeScorer(self._mock_call("Terrible.", "-0.5"))
        r = await scorer.score("Q", "a", "b", pass_threshold=0.7)
        assert r.score == 0.0

    async def test_missing_score_tag_defaults_to_zero(self) -> None:
        scorer = LLMJudgeScorer(lambda _: ("<reasoning>Something.</reasoning>", 0))
        r = await scorer.score("Q", "a", "b", pass_threshold=0.7)
        assert r.score == 0.0

    async def test_missing_reasoning_uses_fallback(self) -> None:
        scorer = LLMJudgeScorer(lambda _: ("<score>0.8</score>", 0))
        r = await scorer.score("Q", "a", "b", pass_threshold=0.7)
        assert r.reasoning == "Could not parse reasoning."

    async def test_non_numeric_score_defaults_to_zero(self) -> None:
        scorer = LLMJudgeScorer(self._mock_call("Hmm.", "high"))
        r = await scorer.score("Q", "a", "b", pass_threshold=0.7)
        assert r.score == 0.0

    async def test_uses_scoring_config_when_present(self) -> None:
        import json
        received: list[str] = []

        def call_claude(prompt: str) -> tuple[str, int]:
            received.append(prompt)
            return "<reasoning>Met criteria.</reasoning>\n<score>1.0</score>", 0

        scorer = LLMJudgeScorer(call_claude)
        config = json.dumps({"criteria": ["Must mention Paris"]})
        r = await scorer.score("Q", "expected", "Paris", pass_threshold=0.7, scoring_config=config)
        assert r.score == 1.0
        assert "Must mention Paris" in received[0]
        # expected_output not used when scoring_config is present
        assert "expected" not in received[0]

    async def test_falls_back_to_expected_without_config(self) -> None:
        received: list[str] = []

        def call_claude(prompt: str) -> tuple[str, int]:
            received.append(prompt)
            return "<reasoning>Good.</reasoning>\n<score>0.8</score>", 0

        scorer = LLMJudgeScorer(call_claude)
        await scorer.score("Q", "expected answer", "actual", pass_threshold=0.7)
        assert "expected answer" in received[0]

    async def test_passed_respects_threshold(self) -> None:
        scorer = LLMJudgeScorer(self._mock_call("Ok.", "0.65"))
        r = await scorer.score("Q", "a", "b", pass_threshold=0.70)
        assert r.passed is False

        scorer2 = LLMJudgeScorer(self._mock_call("Ok.", "0.75"))
        r2 = await scorer2.score("Q", "a", "b", pass_threshold=0.70)
        assert r2.passed is True
