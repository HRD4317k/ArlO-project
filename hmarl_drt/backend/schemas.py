"""Pydantic models for HMARL-DRT FastAPI backend."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime


# ── Enums ─────────────────────────────────────────────
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Config Schemas ───────────────────────────────────
class ConfigSummary(BaseModel):
    name: str
    path: str
    n_episodes: int
    batch_size: int
    device: str


class ConfigUpdate(BaseModel):
    """Partial config update — only include fields to change."""
    n_episodes: Optional[int] = None
    batch_size: Optional[int] = None
    seed: Optional[int] = None
    device: Optional[str] = None
    log_dir: Optional[str] = None
    # Nested overrides
    reward_weights: Optional[Dict[str, float]] = None
    arbiter_temperature: Optional[float] = None
    sac_lr: Optional[float] = None
    sac_gamma: Optional[float] = None


# ── Training Job Schemas ─────────────────────────────
class TrainRequest(BaseModel):
    config_name: str = Field(
        default="test_config",
        description="Config file name (without .yaml extension)",
    )
    overrides: Optional[ConfigUpdate] = None
    run_name: Optional[str] = Field(
        default=None,
        description="Custom run name. Auto-generated if not provided.",
    )


class TrainJob(BaseModel):
    job_id: str
    status: JobStatus
    config_name: str
    run_name: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    current_episode: int = 0
    total_episodes: int = 0
    latest_reward: Optional[float] = None
    error_message: Optional[str] = None
    pid: Optional[int] = None


# ── Run / Metrics Schemas ────────────────────────────
class RunSummary(BaseModel):
    name: str
    log_path: str
    has_checkpoint: bool
    n_episodes: int
    total_steps: int
    mean_reward: Optional[float] = None
    final_phase: Optional[str] = None


class EpisodeMetrics(BaseModel):
    episode: int
    reward: float
    steps: int
    phase: Optional[str] = None
    delta: Optional[float] = None
    rul_estimate: Optional[float] = None
    uncertainty: Optional[float] = None
    physics_violations: Optional[int] = None
    gating_weights: Optional[Dict[str, float]] = None
    reward_components: Optional[Dict[str, float]] = None
    shaper_weights: Optional[Dict[str, float]] = None


class RunMetrics(BaseModel):
    run_name: str
    episodes: List[EpisodeMetrics]
    summary: Dict[str, float]


# ── Simulation Schemas ───────────────────────────────
class SimulationRequest(BaseModel):
    n_steps: int = Field(default=200, ge=1, le=5000)
    seed: Optional[int] = 42
    action_mode: str = Field(
        default="random",
        description="'random', 'zero', or 'constant'",
    )
    constant_action: Optional[List[float]] = None


class SimulationStep(BaseModel):
    step: int
    state: List[float]
    action: List[float]
    reward: float
    done: bool


class SimulationResult(BaseModel):
    n_steps: int
    total_reward: float
    steps: List[SimulationStep]


# ── Generic Responses ────────────────────────────────
class StatusResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    runs_available: int
    active_jobs: int
