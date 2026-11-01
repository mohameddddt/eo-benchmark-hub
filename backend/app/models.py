import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


# UPLOADED -> QUEUED -> VALIDATING -> EVALUATING -> COMPLETED | REJECTED | FAILED
RUNNING_STATUSES = ("UPLOADED", "QUEUED", "VALIDATING", "EVALUATING")


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120))
    affiliation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    website: Mapped[str | None] = mapped_column(String(500), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # user|maintainer|admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    tokens: Mapped[list["ApiToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    submissions: Mapped[list["Submission"]] = relationship(back_populates="user")


class ApiToken(Base):
    __tablename__ = "api_tokens"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    prefix: Mapped[str] = mapped_column(String(12))  # shown in UI: eob_xxxx...
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="tokens")


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    tagline: Mapped[str] = mapped_column(String(255), default="")
    task_type: Mapped[str] = mapped_column(String(40))
    description: Mapped[str] = mapped_column(Text, default="")
    modality: Mapped[str] = mapped_column(String(120), default="")      # "Sentinel-2 RGB · 10 m"
    dataset_url: Mapped[str] = mapped_column(String(500), default="")
    paper_url: Mapped[str] = mapped_column(String(500), default="")
    starter_url: Mapped[str] = mapped_column(String(500), default="")   # Kaggle starter notebook
    citation: Mapped[str] = mapped_column(Text, default="")
    license: Mapped[str] = mapped_column(String(120), default="")
    primary_metric: Mapped[str] = mapped_column(String(40), default="accuracy")
    classes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)    # task-specific (corruptions, …)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)     # split sizes, fingerprint, …
    split_version: Mapped[str] = mapped_column(String(20), default="v2")
    evaluator_version: Mapped[str] = mapped_column(String(60), default="")
    position: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    benchmark_id: Mapped[str] = mapped_column(ForeignKey("benchmarks.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(120), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="UPLOADED", index=True)
    file_uri: Mapped[str] = mapped_column(String(500))
    file_format: Mapped[str] = mapped_column(String(20), default="csv")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    checksum: Mapped[str] = mapped_column(String(64), default="")
    error_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # per-class, confusion, curves…
    evaluator_version: Mapped[str] = mapped_column(String(60), default="")
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)       # reproduced by maintainers
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="submissions")
    benchmark: Mapped[Benchmark] = relationship()
    results: Mapped[list["Result"]] = relationship(back_populates="submission", cascade="all, delete-orphan")
    model_card: Mapped["ModelCard | None"] = relationship(back_populates="submission", uselist=False,
                                                          cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(60))
    value: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)

    submission: Mapped[Submission] = relationship(back_populates="results")


class ModelCard(Base):
    __tablename__ = "model_cards"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"), unique=True)
    architecture: Mapped[str] = mapped_column(String(255), default="")
    family: Mapped[str] = mapped_column(String(60), default="")         # CNN | ViT | ConvNeXt | …
    backbone: Mapped[str] = mapped_column(String(120), default="")
    num_params: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gflops: Mapped[float | None] = mapped_column(Float, nullable=True)
    throughput: Mapped[float | None] = mapped_column(Float, nullable=True)    # images / s
    throughput_setup: Mapped[str] = mapped_column(String(255), default="")
    input_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pretraining: Mapped[str] = mapped_column(String(255), default="")
    external_data: Mapped[bool] = mapped_column(Boolean, default=False)
    epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    optimizer: Mapped[str] = mapped_column(String(255), default="")
    augmentation: Mapped[str] = mapped_column(String(255), default="")
    hardware: Mapped[str] = mapped_column(String(120), default="")
    training_time: Mapped[str] = mapped_column(String(60), default="")
    framework: Mapped[str] = mapped_column(String(120), default="")
    repo_url: Mapped[str] = mapped_column(String(500), default="")
    paper_url: Mapped[str] = mapped_column(String(500), default="")
    notebook_url: Mapped[str] = mapped_column(String(500), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    training_log: Mapped[list | None] = mapped_column(JSON, nullable=True)   # per-epoch curves
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    submission: Mapped[Submission] = relationship(back_populates="model_card")
