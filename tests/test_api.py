from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import aiosqlite
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.database import (
    fetch_all_runs,
    fetch_results_for_run,
    fetch_run_by_id,
    get_db,
    init_db,
    insert_result,
    insert_run,
    update_run_status,
)
from api.main import app, lifespan
from api.routes import _run_eval_background
from api.schemas import RunRequest, TestCaseInput
from eval_runner.models import Result, RunStatus


async def _create_tables(db: aiosqlite.Connection) -> None:
    await db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            llm_model TEXT NOT NULL,
            system_prompt TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        )
    """)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            test_case_input TEXT NOT NULL,
            test_case_expected TEXT NOT NULL,
            actual_output TEXT NOT NULL,
            score REAL NOT NULL,
            reasoning TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        )
    """)
    await db.commit()


@pytest_asyncio.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await _create_tables(conn)
        yield conn


@pytest_asyncio.fixture
async def client(db: aiosqlite.Connection) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[aiosqlite.Connection]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


SAMPLE_REQUEST = {
    "name": "test-run",
    "test_cases": [
        {"input": "What is 2+2?", "expected_output": "4", "scoring_method": "exact_match"}
    ],
}


def make_result(run_id: str) -> Result:
    return Result(
        run_id=uuid4(),
        test_case_id=uuid4(),
        actual_output="4",
        score=1.0,
        reasoning="Exact match.",
        latency_ms=100,
    )


class TestDatabase:
    async def test_init_db_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        with patch("api.database.DB_PATH", db_path):
            await init_db()
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                tables = {row[0] for row in await cursor.fetchall()}
        assert "runs" in tables
        assert "results" in tables

    async def test_get_db_yields_connection(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        with patch("api.database.DB_PATH", db_path):
            gen = get_db()
            conn = await gen.__anext__()
            assert isinstance(conn, aiosqlite.Connection)
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

    async def test_insert_run_returns_id(self, db: aiosqlite.Connection) -> None:
        run_id = await insert_run(db, name="run-1", llm_model="claude-sonnet-4-6", system_prompt="")
        assert run_id is not None

    async def test_update_run_status(self, db: aiosqlite.Connection) -> None:
        run_id = await insert_run(db, name="run-1", llm_model="claude-sonnet-4-6", system_prompt="")
        await update_run_status(db, run_id, RunStatus.completed)
        row = await fetch_run_by_id(db, run_id)
        assert row is not None
        assert row["status"] == RunStatus.completed

    async def test_insert_result_persists(self, db: aiosqlite.Connection) -> None:
        run_id = await insert_run(db, name="run-1", llm_model="claude-sonnet-4-6", system_prompt="")
        await insert_result(
            db,
            run_id=run_id,
            test_case_input="Q?",
            test_case_expected="A",
            actual_output="A",
            score=1.0,
            reasoning="Exact match.",
            latency_ms=100,
            created_at=datetime.now(UTC).isoformat(),
        )
        await db.commit()
        rows = await fetch_results_for_run(db, run_id)
        assert len(rows) == 1
        assert rows[0]["score"] == 1.0

    async def test_fetch_run_by_id_returns_none_for_missing(self, db: aiosqlite.Connection) -> None:
        result = await fetch_run_by_id(db, "nonexistent-id")
        assert result is None

    async def test_fetch_all_runs_returns_list(self, db: aiosqlite.Connection) -> None:
        await insert_run(db, name="run-1", llm_model="claude-sonnet-4-6", system_prompt="")
        await insert_run(db, name="run-2", llm_model="claude-sonnet-4-6", system_prompt="")
        rows = await fetch_all_runs(db)
        assert sum(1 for _ in rows) == 2


class TestLifespan:
    async def test_lifespan_calls_init_db(self) -> None:
        with patch("api.main.init_db") as mock_init:
            async with lifespan(app):
                pass
        mock_init.assert_called_once()


class TestBackgroundTask:
    async def test_inserts_results_on_success(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(db_path) as db:
            await _create_tables(db)
            run_id = str(uuid4())
            await db.execute(
                "INSERT INTO runs (id, name, llm_model, system_prompt, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, "test", "claude-sonnet-4-6", "", RunStatus.running,
                 datetime.now(UTC).isoformat()),
            )
            await db.commit()

        request = RunRequest(
            name="test",
            test_cases=[TestCaseInput(input="What is 2+2?", expected_output="4")],
        )
        mock_result = make_result(run_id)

        with patch("api.routes.DB_PATH", db_path):
            with patch("api.routes.run_eval", return_value=[mock_result]):
                await _run_eval_background(run_id, request)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)) as cursor:
                rows = list(await cursor.fetchall())
            async with db.execute("SELECT status FROM runs WHERE id = ?", (run_id,)) as cursor:
                run = await cursor.fetchone()
        assert len(rows) == 1
        assert run is not None
        assert run["status"] == RunStatus.completed

    async def test_marks_run_failed_on_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(db_path) as db:
            await _create_tables(db)
            run_id = str(uuid4())
            await db.execute(
                "INSERT INTO runs (id, name, llm_model, system_prompt, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, "test", "claude-sonnet-4-6", "", RunStatus.running,
                 datetime.now(UTC).isoformat()),
            )
            await db.commit()

        request = RunRequest(
            name="test",
            test_cases=[TestCaseInput(input="Q?", expected_output="A")],
        )

        with patch("api.routes.DB_PATH", db_path):
            with patch("api.routes.run_eval", side_effect=Exception("API error")):
                await _run_eval_background(run_id, request)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT status FROM runs WHERE id = ?", (run_id,)) as cursor:
                run = await cursor.fetchone()
        assert run is not None
        assert run["status"] == RunStatus.failed


class TestPostRuns:
    async def test_returns_202(self, client: AsyncClient) -> None:
        with patch("api.routes._run_eval_background"):
            response = await client.post("/runs", json=SAMPLE_REQUEST)
        assert response.status_code == 202

    async def test_returns_run_id_and_status(self, client: AsyncClient) -> None:
        with patch("api.routes._run_eval_background"):
            response = await client.post("/runs", json=SAMPLE_REQUEST)
        data = response.json()
        assert "id" in data
        assert data["status"] == RunStatus.running
        assert data["name"] == "test-run"

    async def test_run_appears_in_list(self, client: AsyncClient) -> None:
        with patch("api.routes._run_eval_background"):
            await client.post("/runs", json=SAMPLE_REQUEST)
        response = await client.get("/runs")
        assert len(response.json()) == 1


class TestGetRuns:
    async def test_returns_empty_list_initially(self, client: AsyncClient) -> None:
        response = await client.get("/runs")
        assert response.status_code == 200
        assert response.json() == []

    async def test_returns_run_summary_fields(self, client: AsyncClient) -> None:
        with patch("api.routes._run_eval_background"):
            await client.post("/runs", json=SAMPLE_REQUEST)
        response = await client.get("/runs")
        run = response.json()[0]
        assert "id" in run
        assert "name" in run
        assert "status" in run
        assert "created_at" in run
        assert "result_count" in run


class TestGetRunById:
    async def test_returns_404_for_unknown_id(self, client: AsyncClient) -> None:
        response = await client.get("/runs/nonexistent-id")
        assert response.status_code == 404

    async def test_returns_run_detail(self, client: AsyncClient) -> None:
        with patch("api.routes._run_eval_background"):
            post = await client.post("/runs", json=SAMPLE_REQUEST)
        run_id = post.json()["id"]
        response = await client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == run_id
        assert data["name"] == "test-run"
        assert "results" in data

    async def test_results_populated_after_eval(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        with patch("api.routes._run_eval_background"):
            post = await client.post("/runs", json=SAMPLE_REQUEST)
        run_id = post.json()["id"]

        await db.execute(
            """INSERT INTO results
               (id, run_id, test_case_input, test_case_expected,
                actual_output, score, reasoning, latency_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid4()), run_id, "What is 2+2?", "4", "4",
             1.0, "Exact match.", 100, datetime.now(UTC).isoformat()),
        )
        await db.execute(
            "UPDATE runs SET status = ? WHERE id = ?",
            (RunStatus.completed, run_id),
        )
        await db.commit()

        response = await client.get(f"/runs/{run_id}")
        data = response.json()
        assert data["status"] == RunStatus.completed
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 1.0
