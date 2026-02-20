from unittest.mock import patch

from eval_runner.runner import grade_exact_match, grade_with_llm


class TestGradeExactMatch:
    def test_exact_match_passes(self) -> None:
        score, reasoning = grade_exact_match("Paris", "Paris")
        assert score == 1.0

    def test_exact_match_case_insensitive(self) -> None:
        score, _ = grade_exact_match("paris", "Paris")
        assert score == 1.0

    def test_exact_match_fails_on_mismatch(self) -> None:
        score, _ = grade_exact_match("London", "Paris")
        assert score == 0.0

    def test_exact_match_strips_whitespace(self) -> None:
        score, _ = grade_exact_match("  Paris  ", "Paris")
        assert score == 1.0

    def test_exact_match_fails_on_verbose_response(self) -> None:
        # Claude often returns "The capital of France is Paris." instead of "Paris"
        score, _ = grade_exact_match("The capital of France is Paris.", "Paris")
        assert score == 0.0

    def test_reasoning_on_pass(self) -> None:
        _, reasoning = grade_exact_match("Paris", "Paris")
        assert reasoning == "Exact match."

    def test_reasoning_on_fail_includes_expected_and_actual(self) -> None:
        _, reasoning = grade_exact_match("London", "Paris")
        assert "Paris" in reasoning
        assert "London" in reasoning


class TestGradeWithLlm:
    def _mock_response(self, reasoning: str, score: str) -> str:
        return f"<reasoning>{reasoning}</reasoning>\n<score>{score}</score>"

    def test_parses_score_and_reasoning(self) -> None:
        with patch("eval_runner.runner.call_claude", return_value=(
            self._mock_response("Looks correct.", "0.9"), 0
        )):
            score, reasoning = grade_with_llm("actual", "expected")
        assert score == 0.9
        assert reasoning == "Looks correct."

    def test_clamps_score_above_one(self) -> None:
        with patch("eval_runner.runner.call_claude", return_value=(
            self._mock_response("Perfect.", "1.5"), 0
        )):
            score, _ = grade_with_llm("a", "b")
        assert score == 1.0

    def test_clamps_score_below_zero(self) -> None:
        with patch("eval_runner.runner.call_claude", return_value=(
            self._mock_response("Terrible.", "-0.5"), 0
        )):
            score, _ = grade_with_llm("a", "b")
        assert score == 0.0

    def test_missing_score_tag_defaults_to_zero(self) -> None:
        with patch("eval_runner.runner.call_claude", return_value=(
            "<reasoning>Something.</reasoning>", 0
        )):
            score, _ = grade_with_llm("a", "b")
        assert score == 0.0

    def test_missing_reasoning_tag_uses_fallback(self) -> None:
        with patch("eval_runner.runner.call_claude", return_value=(
            "<score>0.8</score>", 0
        )):
            _, reasoning = grade_with_llm("a", "b")
        assert reasoning == "Could not parse reasoning."

    def test_non_numeric_score_defaults_to_zero(self) -> None:
        with patch("eval_runner.runner.call_claude", return_value=(
            self._mock_response("Hmm.", "high"), 0
        )):
            score, _ = grade_with_llm("a", "b")
        assert score == 0.0
