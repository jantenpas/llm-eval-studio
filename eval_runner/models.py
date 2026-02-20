from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ScoringMethod(StrEnum):
    exact_match = "exact_match"
    llm_judge = "llm_judge"
    fuzzy = "fuzzy"


class RunStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    input: str
    expected_output: str
    scoring_method: ScoringMethod = ScoringMethod.llm_judge
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    name: str
    llm_model: str
    system_prompt: str = ""
    status: RunStatus = RunStatus.pending
    created_at: datetime = Field(default_factory=_utcnow)


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    test_case_id: UUID
    actual_output: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    latency_ms: int
    created_at: datetime = Field(default_factory=_utcnow)
