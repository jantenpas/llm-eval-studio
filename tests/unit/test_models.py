from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from eval_runner.models import Project, Result, Run, RunStatus, ScoringMethod, TestCase


class TestProject:
    def test_creates_with_required_fields(self) -> None:
        p = Project(name="My Project")
        assert p.name == "My Project"

    def test_auto_generates_id(self) -> None:
        p1 = Project(name="A")
        p2 = Project(name="B")
        assert p1.id != p2.id

    def test_description_defaults_to_empty_string(self) -> None:
        p = Project(name="A")
        assert p.description == ""

    def test_created_at_is_timezone_aware(self) -> None:
        p = Project(name="A")
        assert p.created_at.tzinfo is not None

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Project(name="A", unknown_field="x")  # type: ignore[call-arg]


class TestTestCase:
    def test_creates_with_required_fields(self) -> None:
        tc = TestCase(
            project_id=uuid4(),
            input="What is 2 + 2?",
            expected_output="4",
        )
        assert tc.input == "What is 2 + 2?"

    def test_scoring_method_defaults_to_llm_judge(self) -> None:
        tc = TestCase(project_id=uuid4(), input="x", expected_output="y")
        assert tc.scoring_method == ScoringMethod.llm_judge

    def test_tags_default_to_empty_list(self) -> None:
        tc = TestCase(project_id=uuid4(), input="x", expected_output="y")
        assert tc.tags == []

    def test_tags_are_independent_across_instances(self) -> None:
        # Guard against the mutable default argument bug
        tc1 = TestCase(project_id=uuid4(), input="x", expected_output="y")
        tc2 = TestCase(project_id=uuid4(), input="x", expected_output="y")
        tc1.tags.append("test")
        assert tc2.tags == []

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TestCase(project_id=uuid4(), input="x", expected_output="y", unknown="z")  # type: ignore[call-arg]


class TestRun:
    def test_status_defaults_to_pending(self) -> None:
        r = Run(project_id=uuid4(), name="run-1", llm_model="claude-sonnet-4-6")
        assert r.status == RunStatus.pending

    def test_accepts_llm_model_string(self) -> None:
        r = Run(project_id=uuid4(), name="run-1", llm_model="claude-sonnet-4-6")
        assert r.llm_model == "claude-sonnet-4-6"


class TestResult:
    def _valid_result(self, **kwargs: Any) -> Result:
        defaults = dict(
            run_id=uuid4(),
            test_case_id=uuid4(),
            actual_output="some output",
            score=0.8,
            reasoning="looks good",
            latency_ms=500,
        )
        return Result(**{**defaults, **kwargs})

    def test_creates_with_valid_score(self) -> None:
        r = self._valid_result(score=0.8)
        assert r.score == 0.8

    def test_score_at_zero_is_valid(self) -> None:
        r = self._valid_result(score=0.0)
        assert r.score == 0.0

    def test_score_at_one_is_valid(self) -> None:
        r = self._valid_result(score=1.0)
        assert r.score == 1.0

    def test_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_result(score=1.1)

    def test_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_result(score=-0.1)

    def test_invalid_uuid_raises(self) -> None:
        with pytest.raises(ValidationError):
            Result(
                run_id="not-a-uuid",  # type: ignore[arg-type]
                test_case_id=uuid4(),
                actual_output="x",
                score=0.5,
                reasoning="ok",
                latency_ms=100,
            )
