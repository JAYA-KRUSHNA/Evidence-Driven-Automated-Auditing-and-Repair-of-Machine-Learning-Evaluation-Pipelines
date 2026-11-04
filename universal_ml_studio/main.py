"""
Universal AI/ML Studio — Main FastAPI Application.

An Automated Dataset Validation, Preprocessing, Analysis and Machine Learning Platform.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.auditor import router as auditor_router
from .api.datasets import router as datasets_router
from .api.eda import router as eda_router
from .api.experiments import router as experiments_router
from .api.ml import router as ml_router
from .api.models_api import router as models_router
from .api.preprocessing import router as preprocessing_router
from .api.reports import router as reports_router
from .api.validation import router as validation_router
from .database.connection import init_db
from .dataset.samples import SampleDatasetManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("universal_ml_studio")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    logger.info("Initializing database schema...")
    init_db()
    logger.info("Ensuring all built-in sample datasets are present...")
    SampleDatasetManager.ensure_all_samples_exist()
    logger.info("Universal AI/ML Studio is ready.")
    yield
    logger.info("Universal AI/ML Studio is shutting down.")


app = FastAPI(
    title="Universal AI/ML Studio",
    description=(
        "An Automated Dataset Validation, Preprocessing, Analysis and "
        "Machine Learning Platform (Backend REST API)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API Routers ────────────────────────────────────────────────────────
app.include_router(datasets_router)
app.include_router(validation_router)
app.include_router(preprocessing_router)
app.include_router(eda_router)
app.include_router(ml_router)
app.include_router(models_router)
app.include_router(experiments_router)
app.include_router(reports_router)
app.include_router(auditor_router)

# ── Static Files Mount ─────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Health & Dashboard Endpoints ───────────────────────────────────────────────
@app.get("/dashboard", tags=["Dashboard"])
def dashboard():
    """Interactive visual web dashboard for Universal AI/ML Studio."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Dashboard UI file not found"}


@app.get("/", tags=["Landing"])
def root(request: Request):
    """Landing page — served to browsers; JSON for API clients."""
    accept = request.headers.get("accept", "")
    landing_file = STATIC_DIR / "landing.html"
    if "text/html" in accept and landing_file.exists():
        return FileResponse(landing_file)
    return {
        "platform": "Universal AI/ML Studio",
        "version": "1.0.0",
        "status": "online",
        "landing_url": "/",
        "dashboard_url": "/dashboard",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint verifying system components."""
    return {
        "status": "healthy",
        "database": "connected",
        "sample_datasets": len(SampleDatasetManager.list_samples()),
    }
