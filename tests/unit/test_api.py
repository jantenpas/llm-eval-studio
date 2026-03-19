"""
Integration tests for the Phase 2 API.
Covers: projects, test suites, test cases, runs, compare, gate, and migration/DB helpers.
"""
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import aiosqlite
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.database import (
    count_test_cases_for_suite,
    delete_project,
    delete_run,
    delete_suite,
    delete_test_case,
    fetch_all_projects,
    fetch_project_by_id,
    fetch_results_for_run,
    fetch_run_by_id,
    fetch_suite_by_id,
    fetch_suites_for_project,
    fetch_test_case_by_id,
    get_db,
    init_db,
    insert_project,
    insert_result,
    insert_run,
    insert_suite,
    insert_test_case,
    suite_has_runs,
    update_run_completed,
    update_run_failed,
    update_run_started,
    update_test_case,
)
from api.main import app, lifespan
from api.routes import _run_eval_background
from eval_runner.models import RunStatus

# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
    endpoint_url TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS test_suites (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS test_cases (
    id TEXT PRIMARY KEY, suite_id TEXT NOT NULL, input TEXT NOT NULL,
    expected_output TEXT NOT NULL, scoring_config TEXT,
    scoring_method TEXT NOT NULL DEFAULT 'exact_match',
    tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, suite_id TEXT NOT NULL,
    name TEXT NOT NULL, llm_model TEXT NOT NULL, system_prompt TEXT NOT NULL DEFAULT '',
    pass_threshold REAL NOT NULL DEFAULT 0.70, status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT, avg_score REAL, passed INTEGER, created_at TEXT NOT NULL, completed_at TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, test_case_id TEXT NOT NULL,
    input TEXT NOT NULL, expected_output TEXT NOT NULL, scoring_config TEXT,
    actual_output TEXT NOT NULL, scoring_method TEXT NOT NULL,
    score REAL NOT NULL, passed INTEGER NOT NULL, latency_ms INTEGER NOT NULL,
    reasoning TEXT, created_at TEXT NOT NULL
);
"""


@pytest_asyncio.fixture
async def db() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await conn.executescript(SQL_SCHEMA)
        yield conn


@pytest_asyncio.fixture
async def client(db: aiosqlite.Connection) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[aiosqlite.Connection]:
        yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers — seed data
# ---------------------------------------------------------------------------

async def _seed_project(db: aiosqlite.Connection, name: str = "Test Project") -> str:
    return await insert_project(db, name=name, description=None, endpoint_url=None)


async def _seed_suite(db: aiosqlite.Connection, project_id: str, name: str = "Suite A") -> str:
    return await insert_suite(db, project_id=project_id, name=name)


async def _seed_test_case(
    db: aiosqlite.Connection,
    suite_id: str,
    input: str = "What is 2+2?",
    expected: str = "4",
) -> str:
    return await insert_test_case(
        db, suite_id=suite_id, input=input, expected_output=expected,
        scoring_method="exact_match", scoring_config=None, tags="[]",
    )


async def _seed_run(
    db: aiosqlite.Connection,
    project_id: str,
    suite_id: str,
    name: str = "Run 1",
) -> str:
    return await insert_run(
        db, project_id=project_id, suite_id=suite_id, name=name,
        llm_model="claude-sonnet-4-6", system_prompt="", pass_threshold=0.70,
    )


# ---------------------------------------------------------------------------
# DB layer tests
# ---------------------------------------------------------------------------

class TestDatabase:
    async def test_init_db_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        with patch("api.database.DB_PATH", db_path):
            with patch("api.database.MIGRATIONS_DIR", Path(__file__).parent.parent.parent / "migrations"):
                await init_db()
        async with aiosqlite.connect(db_path) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                tables = {row[0] for row in await cursor.fetchall()}
        assert "projects" in tables
        assert "test_suites" in tables
        assert "test_cases" in tables
        assert "runs" in tables
        assert "results" in tables
        assert "schema_migrations" in tables

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

    async def test_insert_and_fetch_project(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        row = await fetch_project_by_id(db, pid)
        assert row is not None
        assert row["name"] == "Test Project"

    async def test_fetch_all_projects_empty(self, db: aiosqlite.Connection) -> None:
        rows = await fetch_all_projects(db)
        assert rows == []

    async def test_delete_project(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        await delete_project(db, pid)
        assert await fetch_project_by_id(db, pid) is None

    async def test_insert_and_fetch_suite(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        row = await fetch_suite_by_id(db, sid)
        assert row is not None
        assert row["name"] == "Suite A"

    async def test_fetch_suites_for_project(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        await _seed_suite(db, pid, "S1")
        await _seed_suite(db, pid, "S2")
        rows = await fetch_suites_for_project(db, pid)
        assert len(rows) == 2

    async def test_suite_has_runs_false(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        assert not await suite_has_runs(db, sid)

    async def test_suite_has_runs_true(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        await _seed_test_case(db, sid)
        await _seed_run(db, pid, sid)
        assert await suite_has_runs(db, sid)

    async def test_delete_suite(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        await delete_suite(db, sid)
        assert await fetch_suite_by_id(db, sid) is None

    async def test_insert_and_fetch_test_case(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        tc_id = await _seed_test_case(db, sid)
        row = await fetch_test_case_by_id(db, tc_id)
        assert row is not None
        assert row["input"] == "What is 2+2?"

    async def test_count_test_cases(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        assert await count_test_cases_for_suite(db, sid) == 0
        await _seed_test_case(db, sid)
        assert await count_test_cases_for_suite(db, sid) == 1

    async def test_update_test_case(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        tc_id = await _seed_test_case(db, sid)
        await update_test_case(
            db, tc_id=tc_id, input="New Q?", expected_output="New A",
            scoring_method="llm_judge", scoring_config=None, tags="[]",
        )
        row = await fetch_test_case_by_id(db, tc_id)
        assert row is not None
        assert row["input"] == "New Q?"
        assert row["scoring_method"] == "llm_judge"

    async def test_delete_test_case(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        tc_id = await _seed_test_case(db, sid)
        await delete_test_case(db, tc_id)
        assert await fetch_test_case_by_id(db, tc_id) is None

    async def test_insert_and_fetch_run(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        run_id = await _seed_run(db, pid, sid)
        row = await fetch_run_by_id(db, run_id)
        assert row is not None
        assert row["status"] == "pending"

    async def test_update_run_started(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        run_id = await _seed_run(db, pid, sid)
        await update_run_started(db, run_id)
        row = await fetch_run_by_id(db, run_id)
        assert row is not None
        assert row["status"] == "running"

    async def test_update_run_completed(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        run_id = await _seed_run(db, pid, sid)
        await update_run_completed(db, run_id, avg_score=0.85, passed=True)
        row = await fetch_run_by_id(db, run_id)
        assert row is not None
        assert row["status"] == "completed"
        assert row["avg_score"] == pytest.approx(0.85)
        assert row["passed"] == 1

    async def test_update_run_failed(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        run_id = await _seed_run(db, pid, sid)
        await update_run_failed(db, run_id, "API error")
        row = await fetch_run_by_id(db, run_id)
        assert row is not None
        assert row["status"] == "failed"
        assert row["error_message"] == "API error"

    async def test_delete_run(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        run_id = await _seed_run(db, pid, sid)
        await delete_run(db, run_id)
        assert await fetch_run_by_id(db, run_id) is None

    async def test_insert_and_fetch_result(self, db: aiosqlite.Connection) -> None:
        pid = await _seed_project(db)
        sid = await _seed_suite(db, pid)
        tc_id = await _seed_test_case(db, sid)
        run_id = await _seed_run(db, pid, sid)
        await insert_result(
            db, run_id=run_id, test_case_id=tc_id, input="Q?",
            expected_output="A", scoring_config=None, actual_output="A",
            scoring_method="exact_match", score=1.0, passed=True,
            latency_ms=100, reasoning="Exact match.",
        )
        await db.commit()
        rows = await fetch_results_for_run(db, run_id)
        assert len(rows) == 1
        assert rows[0]["score"] == 1.0


class TestLifespan:
    async def test_lifespan_calls_init_db(self) -> None:
        with patch("api.main.init_db") as mock_init:
            async with lifespan(app):
                pass
        mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

class TestBackgroundTask:
    async def test_inserts_results_on_success(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(db_path) as conn:
            await conn.executescript(SQL_SCHEMA)
            pid = await insert_project(conn, name="P", description=None, endpoint_url=None)
            sid = await insert_suite(conn, project_id=pid, name="S")
            await insert_test_case(
                conn, suite_id=sid, input="Q?", expected_output="A",
                scoring_method="exact_match", scoring_config=None, tags="[]",
            )
            run_id = await insert_run(
                conn, project_id=pid, suite_id=sid, name="R",
                llm_model="claude-sonnet-4-6", system_prompt="", pass_threshold=0.70,
            )

        mock_scorer = MagicMock()
        mock_scorer.score = AsyncMock(
            return_value=MagicMock(score=1.0, passed=True, reasoning="Exact match.")
        )

        with patch("api.database.DB_PATH", db_path):
            with patch("eval_runner.runner.call_claude", return_value=("A", 100)):
                with patch("eval_runner.scorers.ExactMatchScorer.score", mock_scorer.score):
                    await _run_eval_background(run_id, db_path=db_path)  # type: ignore[arg-type]

        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)) as cursor:
                row = await cursor.fetchone()
        assert row is not None
        assert row["status"] == RunStatus.completed

    async def test_marks_run_failed_on_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        async with aiosqlite.connect(db_path) as conn:
            await conn.executescript(SQL_SCHEMA)
            pid = await insert_project(conn, name="P", description=None, endpoint_url=None)
            sid = await insert_suite(conn, project_id=pid, name="S")
            await insert_test_case(
                conn, suite_id=sid, input="Q?", expected_output="A",
                scoring_method="exact_match", scoring_config=None, tags="[]",
            )
            run_id = await insert_run(
                conn, project_id=pid, suite_id=sid, name="R",
                llm_model="claude-sonnet-4-6", system_prompt="", pass_threshold=0.70,
            )

        with patch("api.database.DB_PATH", db_path):
            with patch("eval_runner.runner.call_claude", side_effect=Exception("API down")):
                await _run_eval_background(run_id, db_path=db_path)  # type: ignore[arg-type]

        async with aiosqlite.connect(db_path) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)) as cursor:
                row = await cursor.fetchone()
        assert row is not None
        assert row["status"] == RunStatus.failed


# ---------------------------------------------------------------------------
# Projects endpoints
# ---------------------------------------------------------------------------

class TestProjects:
    async def test_create_project_201(self, client: AsyncClient) -> None:
        r = await client.post("/projects", json={"name": "My Project"})
        assert r.status_code == 201
        assert r.json()["name"] == "My Project"

    async def test_create_project_400_empty_name(self, client: AsyncClient) -> None:
        r = await client.post("/projects", json={"name": "  "})
        assert r.status_code == 400

    async def test_list_projects_empty(self, client: AsyncClient) -> None:
        r = await client.get("/projects")
        assert r.status_code == 200
        assert r.json() == []

    async def test_list_projects_with_entries(self, client: AsyncClient) -> None:
        await client.post("/projects", json={"name": "P1"})
        await client.post("/projects", json={"name": "P2"})
        r = await client.get("/projects")
        assert len(r.json()) == 2

    async def test_get_project_200(self, client: AsyncClient) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        r = await client.get(f"/projects/{pid}")
        assert r.status_code == 200
        assert r.json()["id"] == pid

    async def test_get_project_404(self, client: AsyncClient) -> None:
        r = await client.get("/projects/nonexistent")
        assert r.status_code == 404

    async def test_delete_project_204(self, client: AsyncClient) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        r = await client.delete(f"/projects/{pid}")
        assert r.status_code == 204

    async def test_delete_project_404(self, client: AsyncClient) -> None:
        r = await client.delete("/projects/nonexistent")
        assert r.status_code == 404

    async def test_delete_project_409_running_run(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        await db.execute(
            "INSERT INTO runs (id, project_id, suite_id, name, llm_model,"
            " system_prompt, pass_threshold, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)",
            (str(uuid4()), pid, sid, "R", "claude-sonnet-4-6", "", 0.70,
             datetime.now(UTC).isoformat()),
        )
        await db.commit()
        r = await client.delete(f"/projects/{pid}")
        assert r.status_code == 409

    async def test_project_detail_includes_suites_and_runs(
        self, client: AsyncClient
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        await client.post(f"/projects/{pid}/suites", json={"name": "S"})
        r = await client.get(f"/projects/{pid}")
        data = r.json()
        assert len(data["suites"]) == 1
        assert data["recent_runs"] == []


# ---------------------------------------------------------------------------
# Test Suites endpoints
# ---------------------------------------------------------------------------

class TestSuites:
    async def test_create_suite_201(self, client: AsyncClient) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        r = await client.post(f"/projects/{pid}/suites", json={"name": "Suite A"})
        assert r.status_code == 201
        assert r.json()["project_id"] == pid

    async def test_create_suite_404_bad_project(self, client: AsyncClient) -> None:
        r = await client.post("/projects/bad/suites", json={"name": "S"})
        assert r.status_code == 404

    async def test_create_suite_400_empty_name(self, client: AsyncClient) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        r = await client.post(f"/projects/{pid}/suites", json={"name": "  "})
        assert r.status_code == 400

    async def test_list_suites_200(self, client: AsyncClient) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        await client.post(f"/projects/{pid}/suites", json={"name": "S1"})
        await client.post(f"/projects/{pid}/suites", json={"name": "S2"})
        r = await client.get(f"/projects/{pid}/suites")
        assert r.status_code == 200
        assert len(r.json()) == 2

    async def test_list_suites_404_bad_project(self, client: AsyncClient) -> None:
        r = await client.get("/projects/bad/suites")
        assert r.status_code == 404

    async def test_delete_suite_204(self, client: AsyncClient) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        r = await client.delete(f"/suites/{sid}")
        assert r.status_code == 204

    async def test_delete_suite_404(self, client: AsyncClient) -> None:
        r = await client.delete("/suites/nonexistent")
        assert r.status_code == 404

    async def test_delete_suite_409_has_runs(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        await db.execute(
            "INSERT INTO runs (id, project_id, suite_id, name, llm_model,"
            " system_prompt, pass_threshold, status, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?)",
            (str(uuid4()), pid, sid, "R", "claude-sonnet-4-6", "", 0.70,
             datetime.now(UTC).isoformat()),
        )
        await db.commit()
        r = await client.delete(f"/suites/{sid}")
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# Test Cases endpoints
# ---------------------------------------------------------------------------

class TestTestCases:
    async def _create_suite(self, client: AsyncClient) -> tuple[str, str]:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        return pid, sid

    async def test_create_test_case_201(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        r = await client.post(
            f"/suites/{sid}/test-cases",
            json={"input": "Q?", "expected_output": "A", "scoring_method": "exact_match"},
        )
        assert r.status_code == 201
        assert r.json()["suite_id"] == sid

    async def test_create_test_case_404_bad_suite(self, client: AsyncClient) -> None:
        r = await client.post(
            "/suites/bad/test-cases",
            json={"input": "Q?", "expected_output": "A", "scoring_method": "exact_match"},
        )
        assert r.status_code == 404

    async def test_create_test_case_422_bad_scoring_method(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        r = await client.post(
            f"/suites/{sid}/test-cases",
            json={"input": "Q?", "expected_output": "A", "scoring_method": "invalid"},
        )
        assert r.status_code == 422

    async def test_list_test_cases(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        await client.post(
            f"/suites/{sid}/test-cases",
            json={"input": "Q1?", "expected_output": "A1", "scoring_method": "exact_match"},
        )
        r = await client.get(f"/suites/{sid}/test-cases")
        assert r.status_code == 200
        assert len(r.json()) == 1

    async def test_list_test_cases_404_bad_suite(self, client: AsyncClient) -> None:
        r = await client.get("/suites/bad/test-cases")
        assert r.status_code == 404

    async def test_update_test_case_200(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        tc_id = (
            await client.post(
                f"/suites/{sid}/test-cases",
                json={"input": "Q?", "expected_output": "A", "scoring_method": "exact_match"},
            )
        ).json()["id"]
        r = await client.put(f"/test-cases/{tc_id}", json={"input": "Updated Q?"})
        assert r.status_code == 200
        assert r.json()["input"] == "Updated Q?"

    async def test_update_test_case_404(self, client: AsyncClient) -> None:
        r = await client.put("/test-cases/nonexistent", json={"input": "X"})
        assert r.status_code == 404

    async def test_delete_test_case_204(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        tc_id = (
            await client.post(
                f"/suites/{sid}/test-cases",
                json={"input": "Q?", "expected_output": "A", "scoring_method": "exact_match"},
            )
        ).json()["id"]
        r = await client.delete(f"/test-cases/{tc_id}")
        assert r.status_code == 204

    async def test_delete_test_case_404(self, client: AsyncClient) -> None:
        r = await client.delete("/test-cases/nonexistent")
        assert r.status_code == 404

    async def test_import_test_cases_201(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        payload = (
            b'[{"input":"Q1","expected_output":"A1","scoring_method":"exact_match"},'
            b'{"input":"Q2","expected_output":"A2","scoring_method":"exact_match"}]'
        )
        r = await client.post(
            f"/suites/{sid}/import",
            files={"file": ("cases.json", payload, "application/json")},
        )
        assert r.status_code == 201
        assert r.json()["imported"] == 2
        assert r.json()["failed"] == 0

    async def test_import_test_cases_400_bad_json(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        r = await client.post(
            f"/suites/{sid}/import",
            files={"file": ("cases.json", b"not json", "application/json")},
        )
        assert r.status_code == 400

    async def test_import_test_cases_400_not_array(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        r = await client.post(
            f"/suites/{sid}/import",
            files={"file": ("cases.json", b'{"key":"value"}', "application/json")},
        )
        assert r.status_code == 400

    async def test_import_counts_failed_rows(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        payload = (
            b'[{"input":"Q1","expected_output":"A1","scoring_method":"exact_match"},'
            b'{"bad":"data"}]'
        )
        r = await client.post(
            f"/suites/{sid}/import",
            files={"file": ("cases.json", payload, "application/json")},
        )
        assert r.json()["imported"] == 1
        assert r.json()["failed"] == 1

    async def test_export_test_cases_200(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        await client.post(
            f"/suites/{sid}/test-cases",
            json={"input": "Q?", "expected_output": "A", "scoring_method": "exact_match"},
        )
        r = await client.get(f"/suites/{sid}/export")
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        data = r.json()
        assert len(data) == 1
        assert data[0]["input"] == "Q?"

    async def test_export_404_bad_suite(self, client: AsyncClient) -> None:
        r = await client.get("/suites/bad/export")
        assert r.status_code == 404

    async def test_scoring_config_roundtrip(self, client: AsyncClient) -> None:
        _, sid = await self._create_suite(client)
        config = {"criteria": ["Must mention Paris", "Tone must be factual"]}
        r = await client.post(
            f"/suites/{sid}/test-cases",
            json={
                "input": "Capital of France?",
                "expected_output": "Paris",
                "scoring_method": "llm_judge",
                "scoring_config": config,
            },
        )
        assert r.status_code == 201
        assert r.json()["scoring_config"] == config


# ---------------------------------------------------------------------------
# Runs endpoints
# ---------------------------------------------------------------------------

class TestRuns:
    async def _setup(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> tuple[str, str, str]:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        await insert_test_case(
            db, suite_id=sid, input="Q?", expected_output="A",
            scoring_method="exact_match", scoring_config=None, tags="[]",
        )
        return pid, sid, sid

    async def test_create_run_202(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid, sid, _ = await self._setup(client, db)
        with patch("api.routes._run_eval_background"):
            r = await client.post("/runs", json={
                "name": "Run 1", "project_id": pid, "suite_id": sid,
                "system_prompt": "You are helpful.",
            })
        assert r.status_code == 202
        assert r.json()["status"] == "pending"

    async def test_create_run_400_empty_name(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid, sid, _ = await self._setup(client, db)
        with patch("api.routes._run_eval_background"):
            r = await client.post("/runs", json={
                "name": "  ", "project_id": pid, "suite_id": sid,
                "system_prompt": "Y",
            })
        assert r.status_code == 400

    async def test_create_run_404_bad_project(self, client: AsyncClient) -> None:
        r = await client.post("/runs", json={
            "name": "R", "project_id": "bad", "suite_id": "bad", "system_prompt": "Y",
        })
        assert r.status_code == 404

    async def test_create_run_409_no_test_cases(
        self, client: AsyncClient
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        r = await client.post("/runs", json={
            "name": "R", "project_id": pid, "suite_id": sid, "system_prompt": "Y",
        })
        assert r.status_code == 409

    async def test_list_runs_200(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid, sid, _ = await self._setup(client, db)
        with patch("api.routes._run_eval_background"):
            await client.post("/runs", json={
                "name": "Run 1", "project_id": pid, "suite_id": sid, "system_prompt": "Y",
            })
        r = await client.get(f"/projects/{pid}/runs")
        assert r.status_code == 200
        assert len(r.json()) == 1

    async def test_list_runs_404_bad_project(self, client: AsyncClient) -> None:
        r = await client.get("/projects/bad/runs")
        assert r.status_code == 404

    async def test_get_run_200(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid, sid, _ = await self._setup(client, db)
        with patch("api.routes._run_eval_background"):
            run_id = (await client.post("/runs", json={
                "name": "R", "project_id": pid, "suite_id": sid, "system_prompt": "Y",
            })).json()["id"]
        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        assert r.json()["id"] == run_id

    async def test_get_run_404(self, client: AsyncClient) -> None:
        r = await client.get("/runs/nonexistent")
        assert r.status_code == 404

    async def test_delete_run_204(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid, sid, _ = await self._setup(client, db)
        with patch("api.routes._run_eval_background"):
            run_id = (await client.post("/runs", json={
                "name": "R", "project_id": pid, "suite_id": sid, "system_prompt": "Y",
            })).json()["id"]
        r = await client.delete(f"/runs/{run_id}")
        assert r.status_code == 204

    async def test_delete_run_409_running(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid, sid, _ = await self._setup(client, db)
        with patch("api.routes._run_eval_background"):
            run_id = (await client.post("/runs", json={
                "name": "R", "project_id": pid, "suite_id": sid, "system_prompt": "Y",
            })).json()["id"]
        await db.execute("UPDATE runs SET status = 'running' WHERE id = ?", (run_id,))
        await db.commit()
        r = await client.delete(f"/runs/{run_id}")
        assert r.status_code == 409

    async def test_delete_run_404(self, client: AsyncClient) -> None:
        r = await client.delete("/runs/nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Compare endpoint
# ---------------------------------------------------------------------------

class TestCompare:
    async def _two_completed_runs(
        self,
        client: AsyncClient,
        db: aiosqlite.Connection,
    ) -> tuple[str, str, str]:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        tc_id = await insert_test_case(
            db, suite_id=sid, input="Q?", expected_output="A",
            scoring_method="exact_match", scoring_config=None, tags="[]",
        )

        run_a = await insert_run(
            db, project_id=pid, suite_id=sid, name="R-A",
            llm_model="claude-sonnet-4-6", system_prompt="", pass_threshold=0.70,
        )
        await db.execute(
            "UPDATE runs SET status='completed', avg_score=0.5, passed=0 WHERE id=?", (run_a,)
        )
        await insert_result(
            db, run_id=run_a, test_case_id=tc_id, input="Q?",
            expected_output="A", scoring_config=None, actual_output="wrong",
            scoring_method="exact_match", score=0.5, passed=False, latency_ms=100, reasoning=None,
        )
        await db.commit()

        run_b = await insert_run(
            db, project_id=pid, suite_id=sid, name="R-B",
            llm_model="claude-sonnet-4-6", system_prompt="", pass_threshold=0.70,
        )
        await db.execute(
            "UPDATE runs SET status='completed', avg_score=1.0, passed=1 WHERE id=?", (run_b,)
        )
        await insert_result(
            db, run_id=run_b, test_case_id=tc_id, input="Q?",
            expected_output="A", scoring_config=None, actual_output="A",
            scoring_method="exact_match", score=1.0, passed=True, latency_ms=80, reasoning=None,
        )
        await db.commit()

        return pid, run_a, run_b

    async def test_compare_200(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        _, run_a, run_b = await self._two_completed_runs(client, db)
        r = await client.get(f"/runs/{run_a}/compare/{run_b}")
        assert r.status_code == 200
        data = r.json()
        assert data["score_delta"] == pytest.approx(0.5)
        assert data["result_count"] == 1
        assert data["results"][0]["change"] == "improved"

    async def test_compare_404_run_not_found(self, client: AsyncClient) -> None:
        r = await client.get("/runs/bad/compare/also-bad")
        assert r.status_code == 404

    async def test_compare_409_different_suites(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid_a = (await client.post(f"/projects/{pid}/suites", json={"name": "SA"})).json()["id"]
        sid_b = (await client.post(f"/projects/{pid}/suites", json={"name": "SB"})).json()["id"]
        run_a = await insert_run(
            db, project_id=pid, suite_id=sid_a, name="RA",
            llm_model="c", system_prompt="", pass_threshold=0.7,
        )
        run_b = await insert_run(
            db, project_id=pid, suite_id=sid_b, name="RB",
            llm_model="c", system_prompt="", pass_threshold=0.7,
        )
        await db.execute("UPDATE runs SET status='completed' WHERE id IN (?, ?)", (run_a, run_b))
        await db.commit()
        r = await client.get(f"/runs/{run_a}/compare/{run_b}")
        assert r.status_code == 409

    async def test_compare_422_not_completed(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        run_a = await insert_run(
            db, project_id=pid, suite_id=sid, name="RA",
            llm_model="c", system_prompt="", pass_threshold=0.7,
        )
        run_b = await insert_run(
            db, project_id=pid, suite_id=sid, name="RB",
            llm_model="c", system_prompt="", pass_threshold=0.7,
        )
        r = await client.get(f"/runs/{run_a}/compare/{run_b}")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Gate endpoint
# ---------------------------------------------------------------------------

class TestGate:
    async def test_gate_200_passed(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        run_id = await insert_run(
            db, project_id=pid, suite_id=sid, name="R",
            llm_model="c", system_prompt="", pass_threshold=0.70,
        )
        await update_run_completed(db, run_id, avg_score=0.85, passed=True)

        r = await client.get(f"/runs/{run_id}/gate")
        assert r.status_code == 200
        data = r.json()
        assert data["passed"] is True
        assert data["score"] == pytest.approx(0.85)
        assert data["threshold"] == pytest.approx(0.70)

    async def test_gate_200_failed(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        run_id = await insert_run(
            db, project_id=pid, suite_id=sid, name="R",
            llm_model="c", system_prompt="", pass_threshold=0.70,
        )
        await update_run_completed(db, run_id, avg_score=0.50, passed=False)

        r = await client.get(f"/runs/{run_id}/gate")
        assert r.status_code == 200
        assert r.json()["passed"] is False

    async def test_gate_404(self, client: AsyncClient) -> None:
        r = await client.get("/runs/nonexistent/gate")
        assert r.status_code == 404

    async def test_gate_422_not_completed(
        self, client: AsyncClient, db: aiosqlite.Connection
    ) -> None:
        pid = (await client.post("/projects", json={"name": "P"})).json()["id"]
        sid = (await client.post(f"/projects/{pid}/suites", json={"name": "S"})).json()["id"]
        run_id = await insert_run(
            db, project_id=pid, suite_id=sid, name="R",
            llm_model="c", system_prompt="", pass_threshold=0.70,
        )
        r = await client.get(f"/runs/{run_id}/gate")
        assert r.status_code == 422
