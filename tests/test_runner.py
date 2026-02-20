import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from anthropic.types import TextBlock
from pydantic import ValidationError

from eval_runner.models import Result, Run, RunStatus, ScoringMethod, TestCase
from eval_runner.runner import (
    call_claude,
    get_client,
    load_test_cases,
    print_summary,
    run_eval,
    run_test_case,
    save_results,
)


def make_anthropic_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[TextBlock(type="text", text=text)])


def make_run(**kwargs: Any) -> Run:
    defaults = dict(project_id=uuid4(), name="test-run", llm_model="claude-sonnet-4-6")
    return Run(**{**defaults, **kwargs})


def make_result(**kwargs: Any) -> Result:
    defaults = dict(
        run_id=uuid4(),
        test_case_id=uuid4(),
        actual_output="output",
        score=1.0,
        reasoning="looks good",
        latency_ms=100,
    )
    return Result(**{**defaults, **kwargs})


class TestGetClient:
    def test_creates_anthropic_client(self) -> None:
        import eval_runner.runner as runner_mod
        runner_mod._client = None
        with patch("eval_runner.runner.load_dotenv") as mock_dotenv:
            with patch("eval_runner.runner.anthropic.Anthropic") as mock_cls:
                client = get_client()
        assert client is mock_cls.return_value
        mock_dotenv.assert_called_once()
        runner_mod._client = None

    def test_caches_client_on_second_call(self) -> None:
        import eval_runner.runner as runner_mod
        runner_mod._client = None
        with patch("eval_runner.runner.load_dotenv"):
            with patch("eval_runner.runner.anthropic.Anthropic") as mock_cls:
                client1 = get_client()
                client2 = get_client()
        assert client1 is client2
        assert mock_cls.call_count == 1
        runner_mod._client = None


class TestCallClaude:
    def test_returns_text_and_latency(self) -> None:
        with patch("eval_runner.runner.get_client") as mock_get_client:
            mock_create = mock_get_client.return_value.messages.create
            mock_create.return_value = make_anthropic_response("Hello!")
            text, latency_ms = call_claude("Say hello")
        assert text == "Hello!"
        assert isinstance(latency_ms, int)
        assert latency_ms >= 0

    def test_includes_system_prompt_when_provided(self) -> None:
        with patch("eval_runner.runner.get_client") as mock_get_client:
            mock_create = mock_get_client.return_value.messages.create
            mock_create.return_value = make_anthropic_response("Hi")
            call_claude("prompt", system_prompt="Be concise.")
            call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "Be concise."

    def test_omits_system_prompt_when_empty(self) -> None:
        with patch("eval_runner.runner.get_client") as mock_get_client:
            mock_create = mock_get_client.return_value.messages.create
            mock_create.return_value = make_anthropic_response("Hi")
            call_claude("prompt")
            call_kwargs = mock_create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_raises_on_empty_content(self) -> None:
        with patch("eval_runner.runner.get_client") as mock_get_client:
            mock_get_client.return_value.messages.create.return_value = SimpleNamespace(content=[])
            with pytest.raises(ValueError, match="empty content list"):
                call_claude("prompt")

    def test_raises_on_non_text_block(self) -> None:
        with patch("eval_runner.runner.get_client") as mock_get_client:
            mock_get_client.return_value.messages.create.return_value = SimpleNamespace(
                content=[SimpleNamespace(type="tool_use")]
            )
            with pytest.raises(ValueError, match="Unexpected content block type"):
                call_claude("prompt")

    def test_accepts_custom_max_tokens(self) -> None:
        with patch("eval_runner.runner.get_client") as mock_get_client:
            mock_create = mock_get_client.return_value.messages.create
            mock_create.return_value = make_anthropic_response("Hi")
            call_claude("prompt", max_tokens=2048)
            call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048


class TestLoadTestCases:
    def test_loads_test_cases_from_json(self, tmp_path: Path) -> None:
        data = [{"input": "What is 2+2?", "expected_output": "4", "scoring_method": "exact_match"}]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(data))
        cases = load_test_cases(path, uuid4())
        assert len(cases) == 1
        assert cases[0].input == "What is 2+2?"

    def test_assigns_project_id(self, tmp_path: Path) -> None:
        project_id = uuid4()
        data = [{"input": "x", "expected_output": "y"}]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(data))
        cases = load_test_cases(path, project_id)
        assert cases[0].project_id == project_id

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_test_cases(Path("nonexistent.json"), uuid4())

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_test_cases(path, uuid4())

    def test_raises_on_non_array_json(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        path.write_text('{"input": "x"}')
        with pytest.raises(ValueError, match="Expected a JSON array"):
            load_test_cases(path, uuid4())

    def test_raises_on_missing_required_fields(self, tmp_path: Path) -> None:
        data = [{"input": "x"}]  # missing expected_output
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            load_test_cases(path, uuid4())

    def test_raises_on_extra_fields(self, tmp_path: Path) -> None:
        data = [{"input": "x", "expected_output": "y", "unknown_field": "z"}]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValidationError):
            load_test_cases(path, uuid4())


class TestRunTestCase:
    def test_exact_match_scoring(self) -> None:
        run = make_run()
        tc = TestCase(
            project_id=run.project_id,
            input="What is 2+2?",
            expected_output="4",
            scoring_method=ScoringMethod.exact_match,
        )
        with patch("eval_runner.runner.call_claude", return_value=("4", 100)):
            result = run_test_case(tc, run)
        assert result.score == 1.0
        assert result.latency_ms == 100

    def test_llm_judge_scoring(self) -> None:
        run = make_run()
        tc = TestCase(
            project_id=run.project_id,
            input="Explain REST.",
            expected_output="A REST API uses HTTP.",
            scoring_method=ScoringMethod.llm_judge,
        )
        with patch("eval_runner.runner.call_claude", side_effect=[
            ("A REST API uses HTTP methods.", 100),
            ("<reasoning>Correct.</reasoning>\n<score>0.9</score>", 50),
        ]):
            result = run_test_case(tc, run)
        assert result.score == 0.9

    def test_raises_for_unimplemented_scoring_method(self) -> None:
        run = make_run()
        tc = TestCase(
            project_id=run.project_id,
            input="x",
            expected_output="y",
            scoring_method=ScoringMethod.fuzzy,
        )
        with patch("eval_runner.runner.call_claude", return_value=("x", 100)):
            with pytest.raises(NotImplementedError):
                run_test_case(tc, run)


class TestPrintSummary:
    def test_shows_pass_for_high_score(self, capsys: pytest.CaptureFixture[str]) -> None:
        run = make_run()
        tc = TestCase(project_id=run.project_id, input="Question?", expected_output="Answer")
        result = make_result(test_case_id=tc.id, score=1.0)
        print_summary([result], [tc])
        assert "PASS" in capsys.readouterr().out

    def test_shows_fail_for_low_score(self, capsys: pytest.CaptureFixture[str]) -> None:
        run = make_run()
        tc = TestCase(project_id=run.project_id, input="Question?", expected_output="Answer")
        result = make_result(test_case_id=tc.id, score=0.5)
        print_summary([result], [tc])
        assert "FAIL" in capsys.readouterr().out

    def test_shows_correct_pass_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        run = make_run()
        tc1 = TestCase(project_id=run.project_id, input="Q1?", expected_output="A1")
        tc2 = TestCase(project_id=run.project_id, input="Q2?", expected_output="A2")
        results = [
            make_result(test_case_id=tc1.id, score=1.0),
            make_result(test_case_id=tc2.id, score=0.5),
        ]
        print_summary(results, [tc1, tc2])
        assert "1/2" in capsys.readouterr().out

    def test_passes_at_threshold(self, capsys: pytest.CaptureFixture[str]) -> None:
        run = make_run()
        tc = TestCase(project_id=run.project_id, input="Question?", expected_output="Answer")
        result = make_result(test_case_id=tc.id, score=0.7)
        print_summary([result], [tc])
        assert "PASS" in capsys.readouterr().out

    def test_fails_just_below_threshold(self, capsys: pytest.CaptureFixture[str]) -> None:
        run = make_run()
        tc = TestCase(project_id=run.project_id, input="Question?", expected_output="Answer")
        result = make_result(test_case_id=tc.id, score=0.69)
        print_summary([result], [tc])
        assert "FAIL" in capsys.readouterr().out

    def test_shows_unknown_for_missing_test_case(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = make_result(score=1.0)
        print_summary([result], [])
        assert "unknown" in capsys.readouterr().out


class TestSaveResults:
    def test_creates_json_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval_runner.runner.RESULTS_DIR", tmp_path)
        run = make_run()
        result = make_result(run_id=run.id)
        save_results([result], run)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].suffix == ".json"

    def test_file_contains_run_and_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eval_runner.runner.RESULTS_DIR", tmp_path)
        run = make_run(name="my-run")
        result = make_result(run_id=run.id)
        save_results([result], run)
        data = json.loads(list(tmp_path.iterdir())[0].read_text())
        assert data["run"]["name"] == "my-run"
        assert len(data["results"]) == 1


class TestRunEval:
    def test_returns_results(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval_runner.runner.RESULTS_DIR", tmp_path / "results")
        cases = [{"input": "What is 2+2?", "expected_output": "4", "scoring_method": "exact_match"}]
        cases_path = tmp_path / "cases.json"
        cases_path.write_text(json.dumps(cases))
        with patch("eval_runner.runner.call_claude", return_value=("4", 100)):
            results = run_eval(test_cases_path=cases_path, run_name="test-run")
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_saves_result_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval_runner.runner.RESULTS_DIR", tmp_path / "results")
        cases = [{"input": "What is 2+2?", "expected_output": "4", "scoring_method": "exact_match"}]
        cases_path = tmp_path / "cases.json"
        cases_path.write_text(json.dumps(cases))
        with patch("eval_runner.runner.call_claude", return_value=("4", 100)):
            run_eval(test_cases_path=cases_path, run_name="test-run")
        result_files = list((tmp_path / "results").iterdir())
        assert len(result_files) == 1
        data = json.loads(result_files[0].read_text())
        assert data["run"]["name"] == "test-run"
        assert len(data["results"]) == 1

    def test_continues_after_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval_runner.runner.RESULTS_DIR", tmp_path / "results")
        cases = [
            {"input": "Q1?", "expected_output": "A1", "scoring_method": "exact_match"},
            {"input": "Q2?", "expected_output": "A2", "scoring_method": "exact_match"},
        ]
        cases_path = tmp_path / "cases.json"
        cases_path.write_text(json.dumps(cases))
        side_effects = [Exception("API error"), ("A2", 100)]
        with patch("eval_runner.runner.call_claude", side_effect=side_effects):
            results = run_eval(test_cases_path=cases_path, run_name="test-run")
        assert len(results) == 1
        assert results[0].actual_output == "A2"
        data = json.loads(list((tmp_path / "results").iterdir())[0].read_text())
        assert data["run"]["status"] == RunStatus.failed

    def test_run_status_is_failed_when_errors_occur(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eval_runner.runner.RESULTS_DIR", tmp_path / "results")
        cases = [{"input": "Q?", "expected_output": "A", "scoring_method": "exact_match"}]
        cases_path = tmp_path / "cases.json"
        cases_path.write_text(json.dumps(cases))
        with patch("eval_runner.runner.call_claude", side_effect=Exception("API error")):
            run_eval(test_cases_path=cases_path, run_name="test-run")
        data = json.loads(list((tmp_path / "results").iterdir())[0].read_text())
        assert data["run"]["status"] == RunStatus.failed
