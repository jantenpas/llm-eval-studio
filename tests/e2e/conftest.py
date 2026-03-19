"""Shared fixtures for e2e tests.

Requires a live server and ANTHROPIC_API_KEY.
Start with: uv run fastapi dev api/main.py
"""
import asyncio
import os

import httpx
import pytest


BASE_URL = os.getenv("EVAL_STUDIO_URL", "http://localhost:8000")
POLL_INTERVAL = 3
POLL_TIMEOUT = 120


@pytest.fixture(scope="session")
def base_url() -> str:
    try:
        httpx.get(f"{BASE_URL}/projects", timeout=2).raise_for_status()
    except Exception:
        pytest.skip("Live server not available — start with: uv run fastapi dev api/main.py")
    return BASE_URL


@pytest.fixture
async def client(base_url: str) -> httpx.AsyncClient:  # type: ignore[misc]
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as c:
        yield c


async def poll_until_complete(client: httpx.AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    """Poll GET /runs/{id} until status is completed or failed."""
    elapsed = 0
    while elapsed < POLL_TIMEOUT:
        r = await client.get(f"/runs/{run_id}")
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("completed", "failed"):
            return data  # type: ignore[no-any-return]
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    pytest.fail(f"Run {run_id} did not complete within {POLL_TIMEOUT}s")
