"""E2E tests for llm_judge scoring with scoring_config rubric — Scenario 3."""
import pytest
import httpx

from tests.e2e.conftest import poll_until_complete

pytestmark = pytest.mark.e2e


class TestLLMJudgeWithRubric:
    """Scenario 3: scoring_config criteria are used instead of expected_output."""

    async def test_rubric_criteria_applied(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E LLM Judge — Rubric"})
        ).raise_for_status().json()["id"]

        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Explanation Quality"})
        ).raise_for_status().json()["id"]

        for tc in [
            {
                "input": "Explain what a database index is.",
                "expected_output": "placeholder",
                "scoring_method": "llm_judge",
                "scoring_config": {
                    "criteria": [
                        "Mentions that an index speeds up queries",
                        "Mentions the trade-off with write speed or storage",
                        "Uses an analogy or concrete example",
                    ]
                },
            },
            {
                "input": "What is the difference between SQL and NoSQL?",
                "expected_output": "placeholder",
                "scoring_method": "llm_judge",
                "scoring_config": {
                    "criteria": [
                        "Mentions schema flexibility",
                        "Mentions scalability differences",
                        "Names at least one example of each type",
                    ]
                },
            },
        ]:
            (await client.post(f"/suites/{suite_id}/test-cases", json=tc)).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Rubric-graded run",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": (
                    "You are a senior software engineer. "
                    "Give clear, accurate technical explanations."
                ),
                "pass_threshold": 0.65,
            })
        ).raise_for_status().json()["id"]

        run = await poll_until_complete(client, run_id)

        assert run["status"] == "completed"
        assert run["passed"] is True

        for result in run["results"]:
            assert result["score"] >= 0.65, f"Score too low for: {result['input']}"
            assert result["reasoning"] is not None
            assert len(result["reasoning"]) > 0

    async def test_each_result_has_reasoning(self, client: httpx.AsyncClient) -> None:
        """reasoning field must be populated for all llm_judge results."""
        project_id = (
            await client.post("/projects", json={"name": "E2E LLM Judge — Reasoning"})
        ).raise_for_status().json()["id"]

        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Reasoning Check"})
        ).raise_for_status().json()["id"]

        (await client.post(f"/suites/{suite_id}/test-cases", json={
            "input": "What is a REST API?",
            "expected_output": "A REST API uses HTTP methods to operate on resources.",
            "scoring_method": "llm_judge",
        })).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Reasoning check run",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "You are a helpful assistant.",
                "pass_threshold": 0.50,
            })
        ).raise_for_status().json()["id"]

        run = await poll_until_complete(client, run_id)

        assert run["status"] == "completed"
        for result in run["results"]:
            assert result["reasoning"] not in (None, "", "Could not parse reasoning.")
