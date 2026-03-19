"""E2E tests for API error paths — Scenario 5."""
import pytest
import httpx

pytestmark = pytest.mark.e2e


class TestNotFound:
    @pytest.mark.parametrize("path", [
        "/runs/does-not-exist",
        "/projects/does-not-exist",
    ])
    async def test_returns_404(self, client: httpx.AsyncClient, path: str) -> None:
        r = await client.get(path)
        assert r.status_code == 404


class TestValidation:
    async def test_invalid_scoring_method_returns_422(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Validation"})
        ).raise_for_status().json()["id"]
        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Suite"})
        ).raise_for_status().json()["id"]

        r = await client.post(f"/suites/{suite_id}/test-cases", json={
            "input": "test",
            "expected_output": "test",
            "scoring_method": "magic",
        })
        assert r.status_code == 422

    async def test_gate_on_incomplete_run_returns_422(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Gate Timing"})
        ).raise_for_status().json()["id"]
        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Suite"})
        ).raise_for_status().json()["id"]
        (await client.post(f"/suites/{suite_id}/test-cases", json={
            "input": "What is the capital of Japan?",
            "expected_output": "Tokyo",
            "scoring_method": "exact_match",
        })).raise_for_status()

        run_id = (
            await client.post("/runs", json={
                "name": "Gate timing",
                "project_id": project_id,
                "suite_id": suite_id,
                "system_prompt": "Answer briefly.",
                "pass_threshold": 0.70,
            })
        ).raise_for_status().json()["id"]

        # Hit gate immediately — run is pending or running
        r = await client.get(f"/runs/{run_id}/gate")
        assert r.status_code == 422


class TestConflicts:
    async def test_run_against_empty_suite_returns_409(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Empty Suite"})
        ).raise_for_status().json()["id"]
        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Empty"})
        ).raise_for_status().json()["id"]

        r = await client.post("/runs", json={
            "name": "Should fail",
            "project_id": project_id,
            "suite_id": suite_id,
            "system_prompt": "test",
            "pass_threshold": 0.70,
        })
        assert r.status_code == 409

    async def test_delete_suite_with_runs_returns_409(self, client: httpx.AsyncClient) -> None:
        project_id = (
            await client.post("/projects", json={"name": "E2E Delete Suite"})
        ).raise_for_status().json()["id"]
        suite_id = (
            await client.post(f"/projects/{project_id}/suites", json={"name": "Suite"})
        ).raise_for_status().json()["id"]
        (await client.post(f"/suites/{suite_id}/test-cases", json={
            "input": "Q", "expected_output": "A", "scoring_method": "exact_match",
        })).raise_for_status()
        (await client.post("/runs", json={
            "name": "Run",
            "project_id": project_id,
            "suite_id": suite_id,
            "system_prompt": "Be concise.",
            "pass_threshold": 0.70,
        })).raise_for_status()

        r = await client.delete(f"/suites/{suite_id}")
        assert r.status_code == 409
