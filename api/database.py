from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

DB_PATH = Path(__file__).parent.parent / "eval_studio.db"
MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        yield db


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        await db.commit()
        await _apply_migrations(db)


async def _apply_migrations(db: aiosqlite.Connection) -> None:
    async with db.execute("SELECT name FROM schema_migrations") as cursor:
        applied = {row["name"] for row in await cursor.fetchall()}

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for path in migration_files:
        if path.name in applied:
            continue
        sql = path.read_text()
        await db.executescript(sql)
        await db.execute(
            "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
            (path.name, datetime.now(UTC).isoformat()),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

async def insert_project(
    db: aiosqlite.Connection,
    name: str,
    description: str | None,
    endpoint_url: str | None,
) -> str:
    project_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO projects (id, name, description, endpoint_url, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (project_id, name, description, endpoint_url, created_at),
    )
    await db.commit()
    return project_id


async def fetch_all_projects(db: aiosqlite.Connection) -> list[aiosqlite.Row]:
    async with db.execute("""
        SELECT
            p.id, p.name, p.description, p.created_at,
            COUNT(r.id) AS run_count,
            (SELECT status FROM runs WHERE project_id = p.id
             ORDER BY created_at DESC LIMIT 1) AS latest_run_status,
            (SELECT avg_score FROM runs WHERE project_id = p.id
             ORDER BY created_at DESC LIMIT 1) AS latest_run_score
        FROM projects p
        LEFT JOIN runs r ON r.project_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """) as cursor:
        return list(await cursor.fetchall())


async def fetch_project_by_id(
    db: aiosqlite.Connection, project_id: str
) -> aiosqlite.Row | None:
    async with db.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,)
    ) as cursor:
        return await cursor.fetchone()


async def delete_project(db: aiosqlite.Connection, project_id: str) -> None:
    await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    await db.commit()


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

async def insert_suite(
    db: aiosqlite.Connection, project_id: str, name: str
) -> str:
    suite_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO test_suites (id, project_id, name, created_at) VALUES (?, ?, ?, ?)",
        (suite_id, project_id, name, created_at),
    )
    await db.commit()
    return suite_id


async def fetch_suites_for_project(
    db: aiosqlite.Connection, project_id: str
) -> list[aiosqlite.Row]:
    async with db.execute("""
        SELECT ts.id, ts.name, ts.created_at, COUNT(tc.id) AS test_case_count
        FROM test_suites ts
        LEFT JOIN test_cases tc ON tc.suite_id = ts.id
        WHERE ts.project_id = ?
        GROUP BY ts.id
        ORDER BY ts.created_at
    """, (project_id,)) as cursor:
        return list(await cursor.fetchall())


async def fetch_suite_by_id(
    db: aiosqlite.Connection, suite_id: str
) -> aiosqlite.Row | None:
    async with db.execute(
        "SELECT * FROM test_suites WHERE id = ?", (suite_id,)
    ) as cursor:
        return await cursor.fetchone()


async def delete_suite(db: aiosqlite.Connection, suite_id: str) -> None:
    await db.execute("DELETE FROM test_suites WHERE id = ?", (suite_id,))
    await db.commit()


async def suite_has_runs(db: aiosqlite.Connection, suite_id: str) -> bool:
    async with db.execute(
        "SELECT 1 FROM runs WHERE suite_id = ? LIMIT 1", (suite_id,)
    ) as cursor:
        return await cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

async def insert_test_case(
    db: aiosqlite.Connection,
    suite_id: str,
    input: str,
    expected_output: str,
    scoring_method: str,
    scoring_config: str | None,
    tags: str,
) -> str:
    tc_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    await db.execute(
        """INSERT INTO test_cases
           (id, suite_id, input, expected_output, scoring_config, scoring_method, tags, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (tc_id, suite_id, input, expected_output, scoring_config, scoring_method, tags, created_at),
    )
    await db.commit()
    return tc_id


async def fetch_test_cases_for_suite(
    db: aiosqlite.Connection, suite_id: str
) -> list[aiosqlite.Row]:
    async with db.execute(
        "SELECT * FROM test_cases WHERE suite_id = ? ORDER BY created_at",
        (suite_id,),
    ) as cursor:
        return list(await cursor.fetchall())


async def fetch_test_case_by_id(
    db: aiosqlite.Connection, tc_id: str
) -> aiosqlite.Row | None:
    async with db.execute(
        "SELECT * FROM test_cases WHERE id = ?", (tc_id,)
    ) as cursor:
        return await cursor.fetchone()


async def update_test_case(
    db: aiosqlite.Connection,
    tc_id: str,
    input: str,
    expected_output: str,
    scoring_method: str,
    scoring_config: str | None,
    tags: str,
) -> None:
    await db.execute(
        """UPDATE test_cases
           SET input = ?, expected_output = ?, scoring_method = ?, scoring_config = ?, tags = ?
           WHERE id = ?""",
        (input, expected_output, scoring_method, scoring_config, tags, tc_id),
    )
    await db.commit()


async def delete_test_case(db: aiosqlite.Connection, tc_id: str) -> None:
    await db.execute("DELETE FROM test_cases WHERE id = ?", (tc_id,))
    await db.commit()


async def count_test_cases_for_suite(
    db: aiosqlite.Connection, suite_id: str
) -> int:
    async with db.execute(
        "SELECT COUNT(*) FROM test_cases WHERE suite_id = ?", (suite_id,)
    ) as cursor:
        row = await cursor.fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

async def insert_run(
    db: aiosqlite.Connection,
    project_id: str,
    suite_id: str,
    name: str,
    llm_model: str,
    system_prompt: str,
    pass_threshold: float,
) -> str:
    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    await db.execute(
        """INSERT INTO runs
           (id, project_id, suite_id, name, llm_model, system_prompt,
            pass_threshold, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
        (run_id, project_id, suite_id, name, llm_model, system_prompt,
         pass_threshold, created_at),
    )
    await db.commit()
    return run_id


async def update_run_started(db: aiosqlite.Connection, run_id: str) -> None:
    await db.execute("UPDATE runs SET status = 'running' WHERE id = ?", (run_id,))
    await db.commit()


async def update_run_completed(
    db: aiosqlite.Connection,
    run_id: str,
    avg_score: float,
    passed: bool,
) -> None:
    completed_at = datetime.now(UTC).isoformat()
    await db.execute(
        """UPDATE runs SET status = 'completed', avg_score = ?, passed = ?, completed_at = ?
           WHERE id = ?""",
        (avg_score, int(passed), completed_at, run_id),
    )
    await db.commit()


async def update_run_failed(
    db: aiosqlite.Connection, run_id: str, error_message: str
) -> None:
    completed_at = datetime.now(UTC).isoformat()
    await db.execute(
        "UPDATE runs SET status = 'failed', error_message = ?, completed_at = ? WHERE id = ?",
        (error_message, completed_at, run_id),
    )
    await db.commit()


async def fetch_run_by_id(
    db: aiosqlite.Connection, run_id: str
) -> aiosqlite.Row | None:
    async with db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)) as cursor:
        return await cursor.fetchone()


async def fetch_runs_for_project(
    db: aiosqlite.Connection, project_id: str
) -> list[aiosqlite.Row]:
    async with db.execute(
        "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ) as cursor:
        return list(await cursor.fetchall())


async def delete_run(db: aiosqlite.Connection, run_id: str) -> None:
    await db.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
    await db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    await db.commit()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

async def insert_result(
    db: aiosqlite.Connection,
    run_id: str,
    test_case_id: str,
    input: str,
    expected_output: str,
    scoring_config: str | None,
    actual_output: str,
    scoring_method: str,
    score: float,
    passed: bool,
    latency_ms: int,
    reasoning: str | None,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    await db.execute(
        """INSERT INTO results
           (id, run_id, test_case_id, input, expected_output, scoring_config,
            actual_output, scoring_method, score, passed, latency_ms, reasoning, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid4()), run_id, test_case_id, input, expected_output, scoring_config,
            actual_output, scoring_method, score, int(passed), latency_ms, reasoning, created_at,
        ),
    )


async def fetch_results_for_run(
    db: aiosqlite.Connection, run_id: str
) -> list[aiosqlite.Row]:
    async with db.execute(
        "SELECT * FROM results WHERE run_id = ? ORDER BY created_at",
        (run_id,),
    ) as cursor:
        return list(await cursor.fetchall())


async def fetch_results_for_runs(
    db: aiosqlite.Connection, run_ids: list[str]
) -> list[aiosqlite.Row]:
    placeholders = ",".join("?" * len(run_ids))
    async with db.execute(
        f"SELECT * FROM results WHERE run_id IN ({placeholders}) ORDER BY run_id, test_case_id",
        run_ids,
    ) as cursor:
        return list(await cursor.fetchall())
