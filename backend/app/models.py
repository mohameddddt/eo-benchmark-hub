import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120))
    affiliation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # user|maintainer|admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    tokens: Mapped[list["ApiToken"]] = relationship(back_populates="user")
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
    task_type: Mapped[str] = mapped_column(String(40))  # scene_classification | ...
    description: Mapped[str] = mapped_column(Text, default="")
    dataset_url: Mapped[str] = mapped_column(String(500), default="")
    license: Mapped[str] = mapped_column(String(120), default="")
    primary_metric: Mapped[str] = mapped_column(String(40), default="accuracy")
    classes: Mapped[list | None] = mapped_column(JSON, nullable=True)
    split_version: Mapped[str] = mapped_column(String(20), default="v1")
    evaluator_version: Mapped[str] = mapped_column(String(60), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    benchmark_id: Mapped[str] = mapped_column(ForeignKey("benchmarks.id"), index=True)
    # UPLOADED -> QUEUED -> VALIDATING -> EVALUATING -> COMPLETED | REJECTED | FAILED
    status: Mapped[str] = mapped_column(String(20), default="UPLOADED", index=True)
    file_uri: Mapped[str] = mapped_column(String(500))
    file_format: Mapped[str] = mapped_column(String(20), default="csv")
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    checksum: Mapped[str] = mapped_column(String(64), default="")
    error_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    evaluator_version: Mapped[str] = mapped_column(String(60), default="")
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="submissions")
    benchmark: Mapped[Benchmark] = relationship()
    results: Mapped[list["Result"]] = relationship(back_populates="submission")
    model_card: Mapped["ModelCard | None"] = relationship(back_populates="submission", uselist=False)


class Result(Base):
    __tablename__ = "results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(60))
    value: Mapped[float] = mapped_column(Float)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # per_class, confusion_matrix

    submission: Mapped[Submission] = relationship(back_populates="results")


class ModelCard(Base):
    __tablename__ = "model_cards"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    submission_id: Mapped[str] = mapped_column(ForeignKey("submissions.id"), unique=True)
    architecture: Mapped[str] = mapped_column(String(255), default="")
    backbone: Mapped[str] = mapped_column(String(120), default="")
    num_params: Mapped[str] = mapped_column(String(40), default="")
    input_size: Mapped[str] = mapped_column(String(40), default="")
    pretraining: Mapped[str] = mapped_column(String(255), default="")
    external_data: Mapped[bool] = mapped_column(Boolean, default=False)
    hardware: Mapped[str] = mapped_column(String(120), default="")
    training_time: Mapped[str] = mapped_column(String(60), default="")
    repo_url: Mapped[str] = mapped_column(String(500), default="")
    paper_url: Mapped[str] = mapped_column(String(500), default="")
    notebook_url: Mapped[str] = mapped_column(String(500), default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    submission: Mapped[Submission] = relationship(back_populates="model_card")
