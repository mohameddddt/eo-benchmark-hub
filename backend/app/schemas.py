from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---- auth ----
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=120)
    affiliation: str | None = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    affiliation: str | None
    role: str

    class Config:
        from_attributes = True


class ApiTokenCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ApiTokenCreatedOut(BaseModel):
    id: str
    name: str
    token: str  # plaintext — shown exactly once
    prefix: str


class ApiTokenOut(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None

    class Config:
        from_attributes = True


# ---- benchmarks ----
class BenchmarkOut(BaseModel):
    slug: str
    name: str
    task_type: str
    description: str
    dataset_url: str
    license: str
    primary_metric: str
    classes: list | None
    split_version: str
    is_active: bool

    class Config:
        from_attributes = True


# ---- submissions ----
class ResultOut(BaseModel):
    metric_name: str
    value: float
    extra: dict | None

    class Config:
        from_attributes = True


class SubmissionOut(BaseModel):
    id: str
    status: str
    file_format: str
    original_filename: str
    checksum: str
    error_report: dict | None
    evaluator_version: str
    is_published: bool
    created_at: datetime
    evaluated_at: datetime | None
    benchmark_slug: str
    results: list[ResultOut] = []


class ModelCardIn(BaseModel):
    architecture: str = ""
    backbone: str = ""
    num_params: str = ""
    input_size: str = ""
    pretraining: str = ""
    external_data: bool = False
    hardware: str = ""
    training_time: str = ""
    repo_url: str = ""
    paper_url: str = ""
    notebook_url: str = ""
    notes: str = ""


class LeaderboardEntry(BaseModel):
    rank: int
    user_name: str
    submission_id: str
    primary_metric: str
    score: float
    created_at: datetime
