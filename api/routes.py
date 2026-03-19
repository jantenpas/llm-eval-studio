import json
import traceback
from pathlib import Path
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.database import (
    DB_PATH,
    count_test_cases_for_suite,
    delete_project,
    delete_run,
    delete_suite,
    delete_test_case,
    fetch_all_projects,
    fetch_project_by_id,
    fetch_results_for_run,
    fetch_results_for_runs,
    fetch_run_by_id,
    fetch_runs_for_project,
    fetch_suite_by_id,
    fetch_suites_for_project,
    fetch_test_case_by_id,
    fetch_test_cases_for_suite,
    get_db,
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
from api.schemas import (
    GateResponse,
    ImportResponse,
    ProjectDetailResponse,
    ProjectListItem,
    ProjectRequest,
    ProjectResponse,
    ResultDelta,
    ResultResponse,
    RunCompareResponse,
    RunCreatedResponse,
    RunDetailResponse,
    RunListItem,
    RunRef,
    RunRequest,
    RunSummary,
    SuiteListItem,
    SuiteRequest,
    SuiteResponse,
    SuiteSummary,
    TestCaseRequest,
    TestCaseResponse,
    TestCaseUpdateRequest,
)
from eval_runner.models import Run, RunStatus, ScoringMethod, TestCase
from eval_runner.runner import run_test_case_async

router = APIRouter()

Db = Annotated[aiosqlite.Connection, Depends(get_db)]


# ---------------------------------------------------------------------------
# Background eval task
# ---------------------------------------------------------------------------

async def _run_eval_background(run_id: str, db_path: Path | None = None) -> None:
    """Execute all test cases for a run and persist results."""
    _db_path = db_path or DB_PATH

    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")
        await update_run_started(db, run_id)

        run_row = await fetch_run_by_id(db, run_id)
        if run_row is None:
            return

        test_case_rows = await fetch_test_cases_for_suite(db, run_row["suite_id"])

    error_message: str | None = None
    results = []

    try:
        run_obj = Run(
            id=run_row["id"],
            project_id=run_row["project_id"],
            name=run_row["name"],
            llm_model=run_row["llm_model"],
            system_prompt=run_row["system_prompt"],
            status=RunStatus.running,
        )
        pass_threshold: float = run_row["pass_threshold"]

        for row in test_case_rows:
            tc = TestCase(
                id=row["id"],
                project_id=run_row["project_id"],
                input=row["input"],
                expected_output=row["expected_output"],
                scoring_method=ScoringMethod(row["scoring_method"]),
                tags=json.loads(row["tags"]),
            )
            result = await run_test_case_async(
                tc, run_obj,
                pass_threshold=pass_threshold,
                scoring_config=row["scoring_config"],
            )
            results.append((tc, row["scoring_config"], result))

    except Exception as exc:
        error_message = traceback.format_exception_only(type(exc), exc)[-1].strip()

    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON")

        if error_message:
            await update_run_failed(db, run_id, error_message)
            return

        for tc, scoring_config, result in results:
            await insert_result(
                db,
                run_id=run_id,
                test_case_id=str(tc.id),
                input=tc.input,
                expected_output=tc.expected_output,
                scoring_config=scoring_config,
                actual_output=result.actual_output,
                scoring_method=str(tc.scoring_method),
                score=result.score,
                passed=result.score >= pass_threshold,
                latency_ms=result.latency_ms,
                reasoning=result.reasoning or None,
            )
        await db.commit()

        if results:
            scores = [r.score for _, _, r in results]
            avg = sum(scores) / len(scores)
            overall_passed = avg >= pass_threshold
        else:
            avg = 0.0
            overall_passed = False

        await update_run_completed(db, run_id, avg_score=avg, passed=overall_passed)


# ---------------------------------------------------------------------------
# Projects — B-5
# ---------------------------------------------------------------------------

@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_project(request: ProjectRequest, db: Db) -> ProjectResponse:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    project_id = await insert_project(
        db,
        name=request.name,
        description=request.description,
        endpoint_url=request.endpoint_url,
    )
    row = await fetch_project_by_id(db, project_id)
    assert row is not None
    return ProjectResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        endpoint_url=row["endpoint_url"],
        created_at=row["created_at"],
    )


@router.get("/projects", response_model=list[ProjectListItem])
async def list_projects(db: Db) -> list[ProjectListItem]:
    rows = await fetch_all_projects(db)
    return [
        ProjectListItem(
            id=r["id"],
            name=r["name"],
            description=r["description"],
            latest_run_status=r["latest_run_status"],
            latest_run_score=r["latest_run_score"],
            run_count=r["run_count"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str, db: Db) -> ProjectDetailResponse:
    row = await fetch_project_by_id(db, project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")

    suite_rows = await fetch_suites_for_project(db, project_id)
    run_rows = await fetch_runs_for_project(db, project_id)

    return ProjectDetailResponse(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        endpoint_url=row["endpoint_url"],
        suites=[
            SuiteSummary(id=s["id"], name=s["name"], test_case_count=s["test_case_count"])
            for s in suite_rows
        ],
        recent_runs=[
            RunSummary(
                id=r["id"],
                name=r["name"],
                status=r["status"],
                avg_score=r["avg_score"],
                passed=bool(r["passed"]) if r["passed"] is not None else None,
                created_at=r["created_at"],
            )
            for r in run_rows[:10]
        ],
        created_at=row["created_at"],
    )


@router.delete("/projects/{project_id}", status_code=204)
async def remove_project(project_id: str, db: Db) -> None:
    row = await fetch_project_by_id(db, project_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")

    run_rows = await fetch_runs_for_project(db, project_id)
    if any(r["status"] == "running" for r in run_rows):
        raise HTTPException(status_code=409, detail="Project has runs currently in progress")

    await delete_project(db, project_id)


# ---------------------------------------------------------------------------
# Test Suites — B-6
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/suites", response_model=SuiteResponse, status_code=201)
async def create_suite(project_id: str, request: SuiteRequest, db: Db) -> SuiteResponse:
    project = await fetch_project_by_id(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    suite_id = await insert_suite(db, project_id=project_id, name=request.name)
    return SuiteResponse(
        id=suite_id,
        project_id=project_id,
        name=request.name,
        created_at=(await fetch_suite_by_id(db, suite_id))["created_at"],  # type: ignore[index]
    )


@router.get("/projects/{project_id}/suites", response_model=list[SuiteListItem])
async def list_suites(project_id: str, db: Db) -> list[SuiteListItem]:
    project = await fetch_project_by_id(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = await fetch_suites_for_project(db, project_id)
    return [
        SuiteListItem(
            id=r["id"],
            name=r["name"],
            test_case_count=r["test_case_count"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


@router.delete("/suites/{suite_id}", status_code=204)
async def remove_suite(suite_id: str, db: Db) -> None:
    suite = await fetch_suite_by_id(db, suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")
    if await suite_has_runs(db, suite_id):
        raise HTTPException(status_code=409, detail="Suite has been used in runs")

    await delete_suite(db, suite_id)


# ---------------------------------------------------------------------------
# Test Cases — B-7
# ---------------------------------------------------------------------------

def _tc_row_to_response(row: aiosqlite.Row) -> TestCaseResponse:
    raw_config = row["scoring_config"]
    scoring_config = json.loads(raw_config) if raw_config else None
    return TestCaseResponse(
        id=row["id"],
        suite_id=row["suite_id"],
        input=row["input"],
        expected_output=row["expected_output"],
        scoring_method=row["scoring_method"],
        scoring_config=scoring_config,
        tags=json.loads(row["tags"]),
        created_at=row["created_at"],
    )


@router.post("/suites/{suite_id}/test-cases", response_model=TestCaseResponse, status_code=201)
async def create_test_case(suite_id: str, request: TestCaseRequest, db: Db) -> TestCaseResponse:
    suite = await fetch_suite_by_id(db, suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")

    scoring_config_json = json.dumps(request.scoring_config) if request.scoring_config else None
    tc_id = await insert_test_case(
        db,
        suite_id=suite_id,
        input=request.input,
        expected_output=request.expected_output,
        scoring_method=request.scoring_method,
        scoring_config=scoring_config_json,
        tags=json.dumps(request.tags),
    )
    row = await fetch_test_case_by_id(db, tc_id)
    assert row is not None
    return _tc_row_to_response(row)


@router.get("/suites/{suite_id}/test-cases", response_model=list[TestCaseResponse])
async def list_test_cases(suite_id: str, db: Db) -> list[TestCaseResponse]:
    suite = await fetch_suite_by_id(db, suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")

    rows = await fetch_test_cases_for_suite(db, suite_id)
    return [_tc_row_to_response(r) for r in rows]


@router.put("/test-cases/{tc_id}", response_model=TestCaseResponse)
async def update_test_case_endpoint(
    tc_id: str, request: TestCaseUpdateRequest, db: Db
) -> TestCaseResponse:
    row = await fetch_test_case_by_id(db, tc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Test case not found")

    new_input = request.input if request.input is not None else row["input"]
    new_expected = (
        request.expected_output if request.expected_output is not None else row["expected_output"]
    )
    new_method = (
        request.scoring_method if request.scoring_method is not None else row["scoring_method"]
    )
    new_config = (
        json.dumps(request.scoring_config) if request.scoring_config is not None
        else row["scoring_config"]
    )
    new_tags = json.dumps(request.tags) if request.tags is not None else row["tags"]

    await update_test_case(
        db,
        tc_id=tc_id,
        input=new_input,
        expected_output=new_expected,
        scoring_method=new_method,
        scoring_config=new_config,
        tags=new_tags,
    )
    updated = await fetch_test_case_by_id(db, tc_id)
    assert updated is not None
    return _tc_row_to_response(updated)


@router.delete("/test-cases/{tc_id}", status_code=204)
async def remove_test_case(tc_id: str, db: Db) -> None:
    row = await fetch_test_case_by_id(db, tc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Test case not found")
    await delete_test_case(db, tc_id)


@router.post("/suites/{suite_id}/import", response_model=ImportResponse, status_code=201)
async def import_test_cases(suite_id: str, file: UploadFile, db: Db) -> ImportResponse:
    suite = await fetch_suite_by_id(db, suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")

    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Expected a JSON array")

    imported = 0
    failed = 0
    for item in data:
        try:
            tc = TestCaseRequest(**item)
            await insert_test_case(
                db,
                suite_id=suite_id,
                input=tc.input,
                expected_output=tc.expected_output,
                scoring_method=tc.scoring_method,
                scoring_config=json.dumps(tc.scoring_config) if tc.scoring_config else None,
                tags=json.dumps(tc.tags),
            )
            imported += 1
        except Exception:
            failed += 1

    return ImportResponse(imported=imported, failed=failed, suite_id=suite_id)


@router.get("/suites/{suite_id}/export")
async def export_test_cases(suite_id: str, db: Db) -> JSONResponse:
    suite = await fetch_suite_by_id(db, suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")

    rows = await fetch_test_cases_for_suite(db, suite_id)
    data = [
        {
            "input": r["input"],
            "expected_output": r["expected_output"],
            "scoring_method": r["scoring_method"],
            "scoring_config": json.loads(r["scoring_config"]) if r["scoring_config"] else None,
            "tags": json.loads(r["tags"]),
        }
        for r in rows
    ]
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f'attachment; filename="suite-{suite_id}.json"'},
    )


# ---------------------------------------------------------------------------
# Runs — B-8
# ---------------------------------------------------------------------------

@router.post("/runs", response_model=RunCreatedResponse, status_code=202)
async def create_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    db: Db,
) -> RunCreatedResponse:
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    project = await fetch_project_by_id(db, request.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    suite = await fetch_suite_by_id(db, request.suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite not found")

    tc_count = await count_test_cases_for_suite(db, request.suite_id)
    if tc_count == 0:
        raise HTTPException(status_code=409, detail="Suite has no test cases")

    run_id = await insert_run(
        db,
        project_id=request.project_id,
        suite_id=request.suite_id,
        name=request.name,
        llm_model=request.llm_model,
        system_prompt=request.system_prompt,
        pass_threshold=request.pass_threshold,
    )

    run_row = await fetch_run_by_id(db, run_id)
    assert run_row is not None
    background_tasks.add_task(_run_eval_background, run_id)

    return RunCreatedResponse(
        id=run_id,
        status="pending",
        created_at=run_row["created_at"],
    )


@router.get("/projects/{project_id}/runs", response_model=list[RunListItem])
async def list_runs(project_id: str, db: Db) -> list[RunListItem]:
    project = await fetch_project_by_id(db, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = await fetch_runs_for_project(db, project_id)
    return [
        RunListItem(
            id=r["id"],
            name=r["name"],
            status=r["status"],
            avg_score=r["avg_score"],
            passed=bool(r["passed"]) if r["passed"] is not None else None,
            pass_threshold=r["pass_threshold"],
            llm_model=r["llm_model"],
            created_at=r["created_at"],
            completed_at=r["completed_at"],
        )
        for r in rows
    ]


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str, db: Db) -> RunDetailResponse:
    run = await fetch_run_by_id(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    result_rows = await fetch_results_for_run(db, run_id)

    return RunDetailResponse(
        id=run["id"],
        name=run["name"],
        project_id=run["project_id"],
        suite_id=run["suite_id"],
        llm_model=run["llm_model"],
        system_prompt=run["system_prompt"],
        pass_threshold=run["pass_threshold"],
        status=run["status"],
        error_message=run["error_message"],
        avg_score=run["avg_score"],
        passed=bool(run["passed"]) if run["passed"] is not None else None,
        created_at=run["created_at"],
        completed_at=run["completed_at"],
        results=[
            ResultResponse(
                id=r["id"],
                test_case_id=r["test_case_id"],
                input=r["input"],
                expected_output=r["expected_output"],
                actual_output=r["actual_output"],
                scoring_method=r["scoring_method"],
                score=r["score"],
                passed=bool(r["passed"]),
                latency_ms=r["latency_ms"],
                reasoning=r["reasoning"],
            )
            for r in result_rows
        ],
    )


@router.delete("/runs/{run_id}", status_code=204)
async def remove_run(run_id: str, db: Db) -> None:
    run = await fetch_run_by_id(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] == "running":
        raise HTTPException(status_code=409, detail="Cannot delete a run in progress")
    await delete_run(db, run_id)


# ---------------------------------------------------------------------------
# Run Comparison — B-9
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/compare/{other_run_id}", response_model=RunCompareResponse)
async def compare_runs(run_id: str, other_run_id: str, db: Db) -> RunCompareResponse:
    run_a = await fetch_run_by_id(db, run_id)
    run_b = await fetch_run_by_id(db, other_run_id)

    if run_a is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if run_b is None:
        raise HTTPException(status_code=404, detail=f"Run {other_run_id} not found")

    if run_a["project_id"] != run_b["project_id"]:
        raise HTTPException(status_code=409, detail="Runs belong to different projects")
    if run_a["suite_id"] != run_b["suite_id"]:
        raise HTTPException(status_code=409, detail="Runs used different test suites")

    if run_a["status"] != "completed" or run_b["status"] != "completed":
        raise HTTPException(status_code=422, detail="Both runs must be completed to compare")

    results_rows = await fetch_results_for_runs(db, [run_id, other_run_id])

    scores_a: dict[str, float] = {}
    scores_b: dict[str, float] = {}
    inputs: dict[str, str] = {}

    for r in results_rows:
        tc_id = r["test_case_id"]
        inputs[tc_id] = r["input"]
        if r["run_id"] == run_id:
            scores_a[tc_id] = r["score"]
        else:
            scores_b[tc_id] = r["score"]

    common_ids = set(scores_a) & set(scores_b)
    deltas: list[ResultDelta] = []
    for tc_id in sorted(common_ids):
        sa, sb = scores_a[tc_id], scores_b[tc_id]
        delta = sb - sa
        if abs(delta) < 0.001:
            change = "unchanged"
        elif delta > 0:
            change = "improved"
        else:
            change = "regressed"
        deltas.append(ResultDelta(
            test_case_id=tc_id,
            input=inputs[tc_id],
            score_a=sa,
            score_b=sb,
            delta=round(delta, 4),
            change=change,
        ))

    avg_a: float | None = run_a["avg_score"]
    avg_b: float | None = run_b["avg_score"]
    score_delta = round(avg_b - avg_a, 4) if avg_a is not None and avg_b is not None else None

    return RunCompareResponse(
        run_a=RunRef(
            id=run_a["id"], name=run_a["name"], avg_score=avg_a, created_at=run_a["created_at"]
        ),
        run_b=RunRef(
            id=run_b["id"], name=run_b["name"], avg_score=avg_b, created_at=run_b["created_at"]
        ),
        score_delta=score_delta,
        result_count=len(deltas),
        results=deltas,
    )


# ---------------------------------------------------------------------------
# Quality Gate — B-10
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/gate", response_model=GateResponse)
async def quality_gate(run_id: str, db: Db) -> GateResponse:
    run = await fetch_run_by_id(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "completed":
        raise HTTPException(status_code=422, detail="Run is not yet completed")

    return GateResponse(
        run_id=run["id"],
        passed=bool(run["passed"]) if run["passed"] is not None else None,
        score=run["avg_score"],
        threshold=run["pass_threshold"],
        status=run["status"],
    )
