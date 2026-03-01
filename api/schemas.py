from pydantic import BaseModel


class TestCaseInput(BaseModel):
    input: str
    expected_output: str
    scoring_method: str = "llm_judge"


class RunRequest(BaseModel):
    name: str
    test_cases: list[TestCaseInput]
    system_prompt: str = ""


class RunCreatedResponse(BaseModel):
    id: str
    name: str
    status: str


class ResultResponse(BaseModel):
    id: str
    test_case_input: str
    test_case_expected: str
    actual_output: str
    score: float
    reasoning: str
    latency_ms: int


class RunSummaryResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: str
    result_count: int
    avg_score: float | None


class RunDetailResponse(BaseModel):
    id: str
    name: str
    llm_model: str
    status: str
    created_at: str
    results: list[ResultResponse]
    total: int
    passed: int
    avg_score: float | None
    avg_latency_ms: float | None
