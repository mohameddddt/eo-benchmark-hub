from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- auth / users ----
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)
    name: str = Field(min_length=1, max_length=120)
    affiliation: str | None = Field(default=None, max_length=255)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(ORM):
    id: str
    email: str
    name: str
    affiliation: str | None
    website: str | None
    bio: str | None
    role: str


class UserUpdateIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    affiliation: str | None = Field(default=None, max_length=255)
    website: str | None = Field(default=None, max_length=500)
    bio: str | None = Field(default=None, max_length=2000)


class UserRef(ORM):
    id: str
    name: str
    affiliation: str | None


class ApiTokenCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ApiTokenCreatedOut(BaseModel):
    id: str
    name: str
    token: str  # plaintext — shown exactly once
    prefix: str


class ApiTokenOut(ORM):
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None


# ---- benchmarks ----
class MetricInfo(BaseModel):
    name: str
    label: str
    description: str
    higher_is_better: bool
    unit: str


class TopEntry(BaseModel):
    submission_id: str
    model_name: str
    score: float


class BenchmarkSummary(BaseModel):
    slug: str
    name: str
    tagline: str
    task_type: str
    modality: str
    primary_metric: str
    primary_metric_label: str
    n_classes: int
    n_entries: int
    ready: bool
    top: TopEntry | None
    position: int


class BenchmarkDetail(BenchmarkSummary):
    description: str
    dataset_url: str
    paper_url: str
    starter_url: str
    citation: str
    license: str
    classes: list | None
    config: dict | None
    stats: dict | None
    split_version: str
    evaluator_version: str
    metrics: list[MetricInfo]
    format_spec: dict
    weekly_quota: int


class BenchmarkRef(BaseModel):
    slug: str
    name: str
    task_type: str
    primary_metric: str


# ---- submissions ----
class ResultOut(ORM):
    metric_name: str
    value: float
    ci_low: float | None
    ci_high: float | None


class ModelCardIn(BaseModel):
    architecture: str = Field(default="", max_length=255)
    family: str = Field(default="", max_length=60)
    backbone: str = Field(default="", max_length=120)
    num_params: int | None = Field(default=None, ge=0)
    gflops: float | None = Field(default=None, ge=0)
    throughput: float | None = Field(default=None, ge=0)
    throughput_setup: str = Field(default="", max_length=255)
    input_size: int | None = Field(default=None, ge=1, le=10000)
    pretraining: str = Field(default="", max_length=255)
    external_data: bool = False
    epochs: int | None = Field(default=None, ge=0)
    optimizer: str = Field(default="", max_length=255)
    augmentation: str = Field(default="", max_length=255)
    hardware: str = Field(default="", max_length=120)
    training_time: str = Field(default="", max_length=60)
    framework: str = Field(default="", max_length=120)
    repo_url: str = Field(default="", max_length=500)
    paper_url: str = Field(default="", max_length=500)
    notebook_url: str = Field(default="", max_length=500)
    notes: str = Field(default="", max_length=5000)


class ModelCardOut(ModelCardIn, ORM):
    training_log: list | None = None
    extra: dict | None = None


class SubmissionUpdateIn(BaseModel):
    model_name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class SubmissionOut(BaseModel):
    id: str
    status: str
    benchmark: BenchmarkRef
    user: UserRef
    model_name: str
    description: str
    file_format: str
    file_size: int
    original_filename: str
    checksum: str
    error_report: dict | None
    evaluator_version: str
    is_published: bool
    verified: bool
    created_at: datetime
    evaluated_at: datetime | None
    results: list[ResultOut] = []
    details: dict | None = None
    model_card: ModelCardOut | None = None
    rank: int | None = None
    n_ranked: int | None = None


class LeaderboardCard(BaseModel):
    architecture: str = ""
    family: str = ""
    pretraining: str = ""
    num_params: int | None = None
    gflops: float | None = None
    throughput: float | None = None
    input_size: int | None = None
    external_data: bool = False
    notebook_url: str = ""


class LeaderboardEntry(BaseModel):
    rank: int
    tier: int
    submission_id: str
    model_name: str
    user: UserRef
    verified: bool
    created_at: datetime
    score: float
    ci_low: float | None
    ci_high: float | None
    metrics: dict[str, float]
    card: LeaderboardCard | None


# ---- site ----
class LeaderHighlight(BaseModel):
    benchmark_slug: str
    benchmark_name: str
    metric_label: str
    model_name: str
    submission_id: str
    score: float
    runner_up_gap: float | None


class RecentEntry(BaseModel):
    submission_id: str
    model_name: str
    benchmark_slug: str
    benchmark_name: str
    user_name: str
    metric_label: str
    score: float
    evaluated_at: datetime | None


class StatsOut(BaseModel):
    benchmarks: int
    task_types: int
    models: int
    evaluations: int
    researchers: int
    leaders: list[LeaderHighlight]
    recent: list[RecentEntry]


class UserPublic(BaseModel):
    id: str
    name: str
    affiliation: str | None
    website: str | None
    bio: str | None
    created_at: datetime
    entries: list[dict]
