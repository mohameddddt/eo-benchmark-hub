from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routes import auth, benchmarks, submissions


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_tables:
        init_db()
    yield


app = FastAPI(
    title="EO-Benchmark Hub API",
    version="0.1.0",
    description="Earth Observation model benchmarking platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(benchmarks.router)
app.include_router(submissions.router)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}
