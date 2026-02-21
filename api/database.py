from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from eval_runner.models import RunStatus

DB_PATH = Path(__file__).parent.parent / "eval_studio.db"


async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
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


async def insert_run(
    db: aiosqlite.Connection,
    name: str,
    llm_model: str,
    system_prompt: str,
) -> str:
    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    await db.execute(
        """
        INSERT INTO runs (id, name, llm_model, system_prompt, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, name, llm_model, system_prompt, RunStatus.running, created_at),
    )
    await db.commit()
    return run_id


async def update_run_status(
    db: aiosqlite.Connection, run_id: str, status: RunStatus
) -> None:
    await db.execute(
        "UPDATE runs SET status = ? WHERE id = ?",
        (status, run_id),
    )
    await db.commit()


async def insert_result(
    db: aiosqlite.Connection,
    run_id: str,
    test_case_input: str,
    test_case_expected: str,
    actual_output: str,
    score: float,
    reasoning: str,
    latency_ms: int,
    created_at: str,
) -> None:
    await db.execute(
        """
        INSERT INTO results
            (id, run_id, test_case_input, test_case_expected,
             actual_output, score, reasoning, latency_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            run_id,
            test_case_input,
            test_case_expected,
            actual_output,
            score,
            reasoning,
            latency_ms,
            created_at,
        ),
    )


async def fetch_run_by_id(
    db: aiosqlite.Connection, run_id: str
) -> aiosqlite.Row | None:
    async with db.execute(
        "SELECT * FROM runs WHERE id = ?", (run_id,)
    ) as cursor:
        return await cursor.fetchone()


async def fetch_all_runs(db: aiosqlite.Connection) -> list[aiosqlite.Row]:
    async with db.execute("""
        SELECT
            r.id, r.name, r.status, r.created_at,
            COUNT(res.id) as result_count,
            AVG(res.score) as avg_score
        FROM runs r
        LEFT JOIN results res ON res.run_id = r.id
        GROUP BY r.id
        ORDER BY r.created_at DESC
    """) as cursor:
        return list(await cursor.fetchall())


async def fetch_results_for_run(
    db: aiosqlite.Connection, run_id: str
) -> list[aiosqlite.Row]:
    async with db.execute(
        "SELECT * FROM results WHERE run_id = ? ORDER BY created_at",
        (run_id,),
    ) as cursor:
        return list(await cursor.fetchall())
