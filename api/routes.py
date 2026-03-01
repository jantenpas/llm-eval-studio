import asyncio
import json
import tempfile
from pathlib import Path
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.database import (
    DB_PATH,
    fetch_all_runs,
    fetch_results_for_run,
    fetch_run_by_id,
    get_db,
    insert_result,
    insert_run,
    update_run_status,
)
from api.schemas import (
    ResultResponse,
    RunCreatedResponse,
    RunDetailResponse,
    RunRequest,
    RunSummaryResponse,
)
from eval_runner.models import RunStatus
from eval_runner.runner import MODEL, PASS_THRESHOLD, run_eval

router = APIRouter()

Db = Annotated[aiosqlite.Connection, Depends(get_db)]


async def _run_eval_background(run_id: str, request: RunRequest) -> None:
    cases = [tc.model_dump() for tc in request.test_cases]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cases, f)
        cases_path = Path(f.name)

    try:
        results = await asyncio.to_thread(
            run_eval,
            test_cases_path=cases_path,
            run_name=request.name,
            system_prompt=request.system_prompt,
        )
        status = RunStatus.completed
    except Exception:
        results = []
        status = RunStatus.failed
    finally:
        cases_path.unlink(missing_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        for i, result in enumerate(results):
            tc = request.test_cases[i]
            await insert_result(
                db,
                run_id=run_id,
                test_case_input=tc.input,
                test_case_expected=tc.expected_output,
                actual_output=result.actual_output,
                score=result.score,
                reasoning=result.reasoning,
                latency_ms=result.latency_ms,
                created_at=result.created_at.isoformat(),
            )
        await update_run_status(db, run_id, status)


@router.post("/runs", response_model=RunCreatedResponse, status_code=202)
async def create_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    db: Db,
) -> RunCreatedResponse:
    run_id = await insert_run(
        db,
        name=request.name,
        llm_model=MODEL,
        system_prompt=request.system_prompt,
    )
    background_tasks.add_task(_run_eval_background, run_id, request)
    return RunCreatedResponse(id=run_id, name=request.name, status=RunStatus.running)


@router.get("/runs", response_model=list[RunSummaryResponse])
async def list_runs(db: Db) -> list[RunSummaryResponse]:
    rows = await fetch_all_runs(db)
    return [
        RunSummaryResponse(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            created_at=row["created_at"],
            result_count=row["result_count"],
            avg_score=row["avg_score"],
        )
        for row in rows
    ]


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str, db: Db) -> RunDetailResponse:
    run = await fetch_run_by_id(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    result_rows = await fetch_results_for_run(db, run_id)

    total = len(result_rows)
    passed = sum(1 for r in result_rows if r["score"] >= PASS_THRESHOLD)
    avg_score = sum(r["score"] for r in result_rows) / total if total else None
    avg_latency_ms = sum(r["latency_ms"] for r in result_rows) / total if total else None

    return RunDetailResponse(
        id=run["id"],
        name=run["name"],
        llm_model=run["llm_model"],
        status=run["status"],
        created_at=run["created_at"],
        results=[
            ResultResponse(
                id=r["id"],
                test_case_input=r["test_case_input"],
                test_case_expected=r["test_case_expected"],
                actual_output=r["actual_output"],
                score=r["score"],
                reasoning=r["reasoning"],
                latency_ms=r["latency_ms"],
            )
            for r in result_rows
        ],
        total=total,
        passed=passed,
        avg_score=avg_score,
        avg_latency_ms=avg_latency_ms,
    )
