"""E2E tests for run comparison and regression detection — Scenario 4."""
import pytest
import httpx

from tests.e2e.conftest import poll_until_complete

pytestmark = pytest.mark.e2e

TEST_CASES = [
    {"input": "What is the capital of Japan?", "expected_output": "Tokyo", "scoring_method": "exact_match"},
    {"input": "What is the capital of Germany?", "expected_output": "Berlin", "scoring_method": "exact_match"},
    {"input": "What is the capital of Australia?", "expected_output": "Canberra", "scoring_method": "exact_match"},
]


async def _create_project_and_suite(client: httpx.AsyncClient, name: str) -> tuple[str, str]:
    project_id = (
        await client.post("/projects", json={"name": name})
    ).raise_for_status().json()["id"]
    suite_id = (
        await client.post(f"/projects/{project_id}/suites", json={"name": "Geography Facts"})
    ).raise_for_status().json()["id"]
    for tc in TEST_CASES:
        (await client.post(f"/suites/{suite_id}/test-cases", json=tc)).raise_for_status()
    return project_id, suite_id


class TestRunComparison:
    """Scenario 4: comparing a precise prompt against a vague one detects regression."""

    async def test_score_delta_is_negative(self, client: httpx.AsyncClient) -> None:
        project_id, suite_id = await _create_project_and_suite(client, "E2E Comparison")

        run_a_id = (
            await client.post("/runs", json={
                "name": "Run A — precise",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "Answer with the city name only. No punctuation.",
                "pass_threshold": 0.70,
            })
        ).raise_for_status().json()["id"]
        await poll_until_complete(client, run_a_id)

        run_b_id = (
            await client.post("/runs", json={
                "name": "Run B — vague",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "You are a helpful assistant.",
                "pass_threshold": 0.70,
            })
        ).raise_for_status().json()["id"]
        await poll_until_complete(client, run_b_id)

        compare = (
            await client.get(f"/runs/{run_a_id}/compare/{run_b_id}")
        ).raise_for_status().json()

        assert compare["score_delta"] < 0, "Expected Run B to score lower than Run A"
        assert compare["result_count"] == len(TEST_CASES)
        assert any(r["change"] == "regressed" for r in compare["results"])

    async def test_compare_response_shape(self, client: httpx.AsyncClient) -> None:
        """Verify all expected fields are present in the compare response."""
        project_id, suite_id = await _create_project_and_suite(client, "E2E Comparison Shape")

        run_a_id = (
            await client.post("/runs", json={
                "name": "Shape check A",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "Answer with the city name only.",
                "pass_threshold": 0.70,
            })
        ).raise_for_status().json()["id"]
        await poll_until_complete(client, run_a_id)

        run_b_id = (
            await client.post("/runs", json={
                "name": "Shape check B",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "You are a helpful assistant.",
                "pass_threshold": 0.70,
            })
        ).raise_for_status().json()["id"]
        await poll_until_complete(client, run_b_id)

        compare = (
            await client.get(f"/runs/{run_a_id}/compare/{run_b_id}")
        ).raise_for_status().json()

        assert "run_a" in compare
        assert "run_b" in compare
        assert "score_delta" in compare
        assert "result_count" in compare
        assert "results" in compare

        for result in compare["results"]:
            assert "test_case_id" in result
            assert "input" in result
            assert "score_a" in result
            assert "score_b" in result
            assert "delta" in result
            assert result["change"] in ("improved", "regressed", "unchanged")
