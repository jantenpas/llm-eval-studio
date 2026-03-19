"""E2E tests for exact_match scoring — Scenarios 1 & 2."""
import pytest
import httpx

from tests.e2e.conftest import poll_until_complete

pytestmark = pytest.mark.e2e


class TestExactMatchConcisePrompt:
    """Scenario 1: concise system prompt should produce exact city name answers."""

    async def test_all_cases_pass(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Exact Match — Concise"})
        ).raise_for_status().json()["id"]

        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Geography Facts"})
        ).raise_for_status().json()["id"]

        for tc in [
            {"input": "What is the capital of Japan?", "expected_output": "Tokyo", "scoring_method": "exact_match"},
            {"input": "What is the capital of Germany?", "expected_output": "Berlin", "scoring_method": "exact_match"},
            {"input": "What is the capital of Australia?", "expected_output": "Canberra", "scoring_method": "exact_match"},
        ]:
            (await client.post(f"/suites/{suite_id}/test-cases", json=tc)).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Concise prompt",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "Answer with the city name only. No punctuation, no sentences.",
                "pass_threshold": 0.80,
            })
        ).raise_for_status().json()["id"]

        run = await poll_until_complete(client, run_id)

        assert run["status"] == "completed"
        assert run["avg_score"] == 1.0
        assert all(r["score"] == 1.0 for r in run["results"])

    async def test_gate_passes(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Gate — Concise"})
        ).raise_for_status().json()["id"]

        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Geography Facts"})
        ).raise_for_status().json()["id"]

        (await client.post(f"/suites/{suite_id}/test-cases", json={
            "input": "What is the capital of Japan?",
            "expected_output": "Tokyo",
            "scoring_method": "exact_match",
        })).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Gate test — concise",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "Answer with the city name only. No punctuation, no sentences.",
                "pass_threshold": 0.80,
            })
        ).raise_for_status().json()["id"]

        await poll_until_complete(client, run_id)

        gate = (await client.get(f"/runs/{run_id}/gate")).raise_for_status().json()
        assert gate["passed"] is True
        assert gate["score"] == 1.0


class TestExactMatchVerbosePrompt:
    """Scenario 2: verbose system prompt causes exact_match failures and trips the gate."""

    async def test_all_cases_fail(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Exact Match — Verbose"})
        ).raise_for_status().json()["id"]

        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Simple Math"})
        ).raise_for_status().json()["id"]

        for tc in [
            {"input": "What is 5 + 3?", "expected_output": "8", "scoring_method": "exact_match"},
            {"input": "What is 10 - 4?", "expected_output": "6", "scoring_method": "exact_match"},
            {"input": "What is 3 × 3?", "expected_output": "9", "scoring_method": "exact_match"},
        ]:
            (await client.post(f"/suites/{suite_id}/test-cases", json=tc)).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Verbose prompt",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": (
                    "You are a math tutor. Always explain your reasoning step by step "
                    "before giving the final answer."
                ),
                "pass_threshold": 0.80,
            })
        ).raise_for_status().json()["id"]

        run = await poll_until_complete(client, run_id)

        assert run["status"] == "completed"
        assert run["avg_score"] == 0.0
        assert all(r["score"] == 0.0 for r in run["results"])

    async def test_gate_fails(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Gate — Verbose"})
        ).raise_for_status().json()["id"]

        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Simple Math"})
        ).raise_for_status().json()["id"]

        (await client.post(f"/suites/{suite_id}/test-cases", json={
            "input": "What is 5 + 3?",
            "expected_output": "8",
            "scoring_method": "exact_match",
        })).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Gate test — verbose",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": (
                    "You are a math tutor. Always explain your reasoning step by step "
                    "before giving the final answer."
                ),
                "pass_threshold": 0.80,
            })
        ).raise_for_status().json()["id"]

        await poll_until_complete(client, run_id)

        gate = (await client.get(f"/runs/{run_id}/gate")).raise_for_status().json()
        assert gate["passed"] is False
