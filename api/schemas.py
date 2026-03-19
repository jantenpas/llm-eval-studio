from pydantic import BaseModel, field_validator

VALID_SCORING_METHODS = {"exact_match", "llm_judge"}


def _validate_scoring_method(v: str) -> str:
    if v not in VALID_SCORING_METHODS:
        raise ValueError(f"scoring_method must be one of {sorted(VALID_SCORING_METHODS)}")
    return v


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class ProjectRequest(BaseModel):
    name: str
    description: str | None = None
    endpoint_url: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str | None
    endpoint_url: str | None
    created_at: str


class ProjectListItem(BaseModel):
    id: str
    name: str
    description: str | None
    latest_run_status: str | None
    latest_run_score: float | None
    run_count: int
    created_at: str


class SuiteSummary(BaseModel):
    id: str
    name: str
    test_case_count: int


class RunSummary(BaseModel):
    id: str
    name: str
    status: str
    avg_score: float | None
    passed: bool | None
    created_at: str


class ProjectDetailResponse(BaseModel):
    id: str
    name: str
    description: str | None
    endpoint_url: str | None
    suites: list[SuiteSummary]
    recent_runs: list[RunSummary]
    created_at: str


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

class SuiteRequest(BaseModel):
    name: str


class SuiteResponse(BaseModel):
    id: str
    project_id: str
    name: str
    created_at: str


class SuiteListItem(BaseModel):
    id: str
    name: str
    test_case_count: int
    created_at: str


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestCaseRequest(BaseModel):
    input: str
    expected_output: str
    scoring_method: str = "exact_match"
    scoring_config: dict[str, list[str]] | None = None
    tags: list[str] = []

    @field_validator("scoring_method")
    @classmethod
    def check_scoring_method(cls, v: str) -> str:
        return _validate_scoring_method(v)


class TestCaseUpdateRequest(BaseModel):
    input: str | None = None
    expected_output: str | None = None
    scoring_method: str | None = None
    scoring_config: dict[str, list[str]] | None = None
    tags: list[str] | None = None

    @field_validator("scoring_method")
    @classmethod
    def check_scoring_method(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _validate_scoring_method(v)


class TestCaseResponse(BaseModel):
    id: str
    suite_id: str
    input: str
    expected_output: str
    scoring_method: str
    scoring_config: dict[str, list[str]] | None
    tags: list[str]
    created_at: str


class ImportResponse(BaseModel):
    imported: int
    failed: int
    suite_id: str


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    name: str
    project_id: str
    suite_id: str
    system_prompt: str
    llm_model: str = "claude-sonnet-4-6"
    pass_threshold: float = 0.70


class RunCreatedResponse(BaseModel):
    id: str
    status: str
    created_at: str


class ResultResponse(BaseModel):
    id: str
    test_case_id: str
    input: str
    expected_output: str
    actual_output: str
    scoring_method: str
    score: float
    passed: bool
    latency_ms: int
    reasoning: str | None


class RunDetailResponse(BaseModel):
    id: str
    name: str
    project_id: str
    suite_id: str
    llm_model: str
    system_prompt: str
    pass_threshold: float
    status: str
    error_message: str | None
    avg_score: float | None
    passed: bool | None
    created_at: str
    completed_at: str | None
    results: list[ResultResponse]


class RunListItem(BaseModel):
    id: str
    name: str
    status: str
    avg_score: float | None
    passed: bool | None
    pass_threshold: float
    llm_model: str
    created_at: str
    completed_at: str | None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class RunRef(BaseModel):
    id: str
    name: str
    avg_score: float | None
    created_at: str


class ResultDelta(BaseModel):
    test_case_id: str
    input: str
    score_a: float
    score_b: float
    delta: float
    change: str  # "improved" | "regressed" | "unchanged"


class RunCompareResponse(BaseModel):
    run_a: RunRef
    run_b: RunRef
    score_delta: float | None
    result_count: int
    results: list[ResultDelta]


# ---------------------------------------------------------------------------
# Quality Gate
# ---------------------------------------------------------------------------

class GateResponse(BaseModel):
    run_id: str
    passed: bool | None
    score: float | None
    threshold: float
    status: str
