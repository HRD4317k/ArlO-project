"""
Training Job Manager — manages background training processes.

Each training job runs train.py as a subprocess and streams its stdout
to a log file. The manager tracks status via a shared dict + file I/O.
"""

import os
import sys
import uuid
import json
import signal
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from backend.schemas import JobStatus, TrainJob

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TrainingJobManager:
    """Singleton manager for training job lifecycle."""

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._jobs_dir = PROJECT_ROOT / "runs" / ".jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        # Restore persisted jobs on startup
        self._restore_jobs()

    # ── Public API ───────────────────────────────────
    def create_job(
        self,
        config_name: str,
        run_name: Optional[str] = None,
        overrides_path: Optional[str] = None,
    ) -> TrainJob:
        job_id = uuid.uuid4().hex[:12]
        if not run_name:
            run_name = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{job_id[:6]}"

        job = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "config_name": config_name,
            "run_name": run_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "finished_at": None,
            "current_episode": 0,
            "total_episodes": 0,
            "latest_reward": None,
            "error_message": None,
            "pid": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._persist_job(job)
        return TrainJob(**job)

    def start_job(self, job_id: str) -> TrainJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            if job["status"] not in (JobStatus.QUEUED,):
                raise ValueError(f"Job {job_id} cannot be started (status={job['status']})")

        config_path = PROJECT_ROOT / "configs" / f"{job['config_name']}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Read total_episodes from config
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        job["total_episodes"] = cfg.get("training", {}).get("n_episodes", 0)

        # Prepare log directory
        run_dir = PROJECT_ROOT / "runs" / job["run_name"]
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable, "-u",
            str(PROJECT_ROOT / "train.py"),
            str(config_path),
        ]

        # Set log_dir override via env
        env = os.environ.copy()
        env["HMARL_LOG_DIR"] = str(run_dir)
        env["PYTHONUNBUFFERED"] = "1"

        stdout_log = run_dir / "train_stdout.log"
        stderr_log = run_dir / "train_stderr.log"

        stdout_fh = open(stdout_log, "w")
        stderr_fh = open(stderr_log, "w")

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=env,
            start_new_session=True,
        )

        with self._lock:
            job["status"] = JobStatus.RUNNING
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            job["pid"] = proc.pid
            self._processes[job_id] = proc

        self._persist_job(job)

        # Launch monitor thread
        t = threading.Thread(
            target=self._monitor_process,
            args=(job_id, proc, stdout_fh, stderr_fh),
            daemon=True,
        )
        t.start()

        return TrainJob(**job)

    def cancel_job(self, job_id: str) -> TrainJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")
            proc = self._processes.get(job_id)

        if proc and proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        with self._lock:
            job["status"] = JobStatus.CANCELLED
            job["finished_at"] = datetime.now(timezone.utc).isoformat()

        self._persist_job(job)
        return TrainJob(**job)

    def get_job(self, job_id: str) -> Optional[TrainJob]:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return None
        # Refresh episode progress from CSV
        self._refresh_progress(job)
        return TrainJob(**job)

    def list_jobs(self) -> list:
        with self._lock:
            jobs = list(self._jobs.values())
        for j in jobs:
            self._refresh_progress(j)
        return [TrainJob(**j) for j in jobs]

    def get_active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["status"] == JobStatus.RUNNING)

    def get_job_logs(self, job_id: str, tail: int = 50) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return ""
        log_path = PROJECT_ROOT / "runs" / job["run_name"] / "train_stdout.log"
        if not log_path.exists():
            return ""
        lines = log_path.read_text().splitlines()
        return "\n".join(lines[-tail:])

    # ── Internal ─────────────────────────────────────
    def _monitor_process(
        self,
        job_id: str,
        proc: subprocess.Popen,
        stdout_fh,
        stderr_fh,
    ):
        """Wait for process to finish and update status."""
        proc.wait()
        stdout_fh.close()
        stderr_fh.close()

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if job["status"] == JobStatus.CANCELLED:
                return
            if proc.returncode == 0:
                job["status"] = JobStatus.COMPLETED
            else:
                job["status"] = JobStatus.FAILED
                stderr_path = PROJECT_ROOT / "runs" / job["run_name"] / "train_stderr.log"
                if stderr_path.exists():
                    err_lines = stderr_path.read_text().splitlines()
                    job["error_message"] = "\n".join(err_lines[-10:])
            job["finished_at"] = datetime.now(timezone.utc).isoformat()

        self._persist_job(job)

    def _refresh_progress(self, job: dict):
        """Read training_log.csv to get current episode & latest reward."""
        csv_path = PROJECT_ROOT / "runs" / job["run_name"] / "training_log.csv"
        if not csv_path.exists():
            return
        try:
            # Read only last line efficiently
            with open(csv_path, "r") as f:
                header = f.readline().strip().split(",")
                last_line = None
                count = 0
                for line in f:
                    if line.strip():
                        last_line = line.strip()
                        count += 1
            if last_line and count > 0:
                values = last_line.split(",")
                if "episode" in header:
                    idx = header.index("episode")
                    if idx < len(values):
                        try:
                            job["current_episode"] = int(float(values[idx])) + 1
                        except ValueError:
                            pass
                if "reward" in header:
                    idx = header.index("reward")
                    if idx < len(values):
                        try:
                            job["latest_reward"] = float(values[idx])
                        except ValueError:
                            pass
        except Exception:
            pass

    def _persist_job(self, job: dict):
        """Save job metadata to disk."""
        path = self._jobs_dir / f"{job['job_id']}.json"
        serializable = dict(job)
        # Convert enum to string for JSON
        if isinstance(serializable.get("status"), JobStatus):
            serializable["status"] = serializable["status"].value
        path.write_text(json.dumps(serializable, indent=2))

    def _restore_jobs(self):
        """Load persisted jobs on startup."""
        for f in self._jobs_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                # Convert status string back to enum
                data["status"] = JobStatus(data["status"])
                # Mark any "running" jobs as failed (stale from crash)
                if data["status"] == JobStatus.RUNNING:
                    data["status"] = JobStatus.FAILED
                    data["error_message"] = "Server restarted — job was interrupted"
                    data["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._jobs[data["job_id"]] = data
            except Exception:
                pass


# Global singleton
job_manager = TrainingJobManager()
