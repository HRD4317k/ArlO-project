"""
HMARL-DRT FastAPI Backend
=========================
Provides REST API for:
 - Training job management (start / stop / status / logs)
 - Config management (list / view / update / clone)
 - Run metrics & data (browse / compare / download)
 - Environment simulation (interactive rollouts)

Run:
    cd hmarl_drt && uvicorn backend.main:app --reload --port 8000
"""

import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HealthResponse
from backend.job_manager import job_manager
from backend.routes import configs, training, runs, simulation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "runs"

# ── App ──────────────────────────────────────────────
app = FastAPI(
    title="HMARL-DRT API",
    description="Backend for Heterogeneous Multi-Agent RL with Dynamic Reward Shaping & Transfer Learning",
    version="1.0.0",
)

# CORS — allow Streamlit (8501) and dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501",
                    "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ────────────────────────────────
app.include_router(configs.router)
app.include_router(training.router)
app.include_router(runs.router)
app.include_router(simulation.router)


# ── Health ───────────────────────────────────────────
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.get("/api/health", response_model=HealthResponse)
def health():
    n_runs = sum(1 for _ in RUNS_DIR.rglob("training_log.csv")) if RUNS_DIR.exists() else 0
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=_detect_device(),
        runs_available=n_runs,
        active_jobs=job_manager.get_active_count(),
    )


# ── Workflow Info ────────────────────────────────────
@app.get("/api/workflow")
def workflow():
    """Return the recommended user workflow steps."""
    return {
        "workflow": [
            {
                "step": 1,
                "title": "Configure Training",
                "description": "Browse or edit training configs (episodes, batch size, reward weights, etc.)",
                "endpoints": ["GET /api/configs/", "GET /api/configs/{name}", "PUT /api/configs/{name}"],
            },
            {
                "step": 2,
                "title": "Start Training",
                "description": "Launch a training job with a chosen config. Returns a job_id for tracking.",
                "endpoints": ["POST /api/training/start"],
            },
            {
                "step": 3,
                "title": "Monitor Training",
                "description": "Poll job status, view live logs, and track episode progress.",
                "endpoints": ["GET /api/training/jobs/{id}", "GET /api/training/jobs/{id}/logs"],
            },
            {
                "step": 4,
                "title": "Analyse Results",
                "description": "Browse completed runs, view episode metrics, compare runs, download data.",
                "endpoints": ["GET /api/runs/", "GET /api/runs/{name}", "GET /api/runs/{a}/compare/{b}"],
            },
            {
                "step": 5,
                "title": "Simulate Environment",
                "description": "Run the WindTurbine environment interactively without training.",
                "endpoints": ["POST /api/simulation/run"],
            },
            {
                "step": 6,
                "title": "Export Artefacts",
                "description": "Download training CSV logs or model checkpoints.",
                "endpoints": ["GET /api/runs/{name}/csv", "GET /api/runs/{name}/checkpoint"],
            },
        ]
    }
