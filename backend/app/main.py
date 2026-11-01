import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.database import init_db
from app.routes import auth, benchmarks, site, submissions
from app.services.runner import recover_pending

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("eobench")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_tables:
        init_db()
    if settings.jwt_secret.startswith("dev-secret"):
        log.warning("JWT_SECRET is the development default — set it before exposing this server")
    recover_pending()
    yield


app = FastAPI(
    title="EO-Bench Hub API",
    version="2.0.0",
    description="Earth-observation model benchmarking: hidden-split evaluation, "
                "CI-aware leaderboards, robustness and efficiency tracking.",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

for r in (auth.router, benchmarks.router, submissions.router, site.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health", tags=["ops"])
def health():
    return {"status": "ok", "eval_backend": settings.eval_backend}


# ---- the built React app (npm run build) is served from the same origin ----
DIST = Path(settings.frontend_dist)


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "Not found")
    if not (DIST / "index.html").exists():
        raise HTTPException(404, "Frontend not built — run `npm run build` in frontend/ "
                                 "or use the Vite dev server on :5173")
    target = (DIST / path).resolve()
    if path and target.is_file() and DIST.resolve() in target.parents:
        return FileResponse(target)
    return FileResponse(DIST / "index.html")  # client-side routing
