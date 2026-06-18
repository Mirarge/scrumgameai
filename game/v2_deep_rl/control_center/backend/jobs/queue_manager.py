from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from services.app_paths import BACKEND_DIR, ensure_engine_import_path
from services.io_utils import safe_float, safe_int, tail_csv_rows
from jobs.processes import choose_python_command, is_pid_running
from jobs.run_paths import create_job_run_dir, default_stdout_log
from storage.jobs_db import create_job, delete_job, get_job, init_db, list_jobs as list_jobs_db, update_job, utc_now_iso

ensure_engine_import_path()


# Throttle refresh_job_states() so concurrent requests don't all hammer SQLite
# with PID checks and writes simultaneously.
_REFRESH_LOCK = threading.Lock()
_LAST_REFRESH_TIME: float = 0.0
_REFRESH_INTERVAL: float = 3.0  # seconds


def refresh_job_states() -> list[dict]:
    """Check running job PIDs and mark dead ones as failed.

    Throttled to run at most once every _REFRESH_INTERVAL seconds so that
    concurrent frontend requests (detail + progress + log) don't all race to
    hammer SQLite with PID checks and writes simultaneously.
    """
    global _LAST_REFRESH_TIME
    init_db()
    now = time.monotonic()
    with _REFRESH_LOCK:
        if now - _LAST_REFRESH_TIME < _REFRESH_INTERVAL:
            # Another request refreshed recently — just return current DB state.
            return list_jobs_db()
        _LAST_REFRESH_TIME = now

    jobs = list_jobs_db()
    for job in jobs:
        if job["status"] == "running" and not is_pid_running(job.get("worker_pid")):
            update_job(
                job["id"],
                status="failed",
                ended_at=utc_now_iso(),
                error_message=job.get("error_message") or "Worker process exited unexpectedly.",
            )
    return list_jobs_db()


def dispatch_next_job() -> dict | None:
    init_db()
    jobs = refresh_job_states()
    if any(job["status"] == "running" for job in jobs):
        return None

    queued_job = next((job for job in reversed(jobs) if job["status"] == "queued"), None)
    if queued_job is None:
        return None

    stdout_path = queued_job.get("stdout_log_path") or ""
    stdout_log_path = Path(stdout_path) if stdout_path else default_stdout_log(Path(queued_job["run_dir"]), queued_job["job_type"])
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_log_path.open("ab") as stdout_handle:
        popen_kwargs = {
            "cwd": BACKEND_DIR,
            "stdout": stdout_handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            [
                choose_python_command(),
                str(BACKEND_DIR / "jobs" / "job_runner.py"),
                "--job-id",
                str(queued_job["id"]),
            ],
            **popen_kwargs,
        )

    return update_job(
        queued_job["id"],
        status="running",
        started_at=utc_now_iso(),
        worker_pid=process.pid,
        stdout_log_path=str(stdout_log_path),
        error_message=None,
    )


def list_jobs() -> list[dict]:
    init_db()
    return refresh_job_states()


def get_job_details(job_id: int) -> dict | None:
    init_db()
    return get_job(job_id)


def _read_run_metadata(run_dir: Path | None) -> dict:
    if not run_dir:
        return {}
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def get_job_progress(job_id: int) -> dict | None:
    init_db()
    refresh_job_states()
    job = get_job(job_id)
    if job is None:
        return None

    run_dir_raw = job.get("run_dir")
    run_dir = Path(run_dir_raw) if run_dir_raw else None
    payload = job.get("payload", {})
    run_metadata = _read_run_metadata(run_dir)
    total_episodes = safe_int(str(run_metadata.get("episodes_this_run", "")))
    if total_episodes is None:
        total_episodes = safe_int(str(payload.get("episodes", "")))
    start_episode = safe_int(str(run_metadata.get("start_episode", ""))) or 1

    progress = {
        "job_id": job["id"],
        "job_type": job["job_type"],
        "status": job["status"],
        "run_dir": str(run_dir) if run_dir else "",
        "stdout_log_path": job.get("stdout_log_path") or "",
        "error_message": job.get("error_message"),
        "total_episodes": total_episodes,
        "start_episode": start_episode,
        "end_episode": safe_int(str(run_metadata.get("end_episode", ""))),
        "latest_episode": 0,
        "completed_episodes": 0,
        "progress_ratio": 0.0,
        "latest_training_row": None,
        "latest_evaluation_row": None,
        "training_series": [],
        "evaluation_series": [],
    }

    if not run_dir:
        return progress

    reports_dir = run_dir / "reports"
    training_rows = tail_csv_rows(reports_dir / "logs.csv", limit=240)
    evaluation_rows = tail_csv_rows(reports_dir / "evaluation_history.csv", limit=120)

    training_series = []
    for row in training_rows:
        episode = safe_int(row.get("episode"))
        if episode is None:
            continue
        training_series.append(
            {
                "episode": episode,
                "episode_reward": safe_float(row.get("episode_reward")),
                "rolling_average_reward": safe_float(row.get("rolling_average_reward")),
                "mean_recent_loss": safe_float(row.get("mean_recent_loss")),
                "average_ending_money": safe_float(row.get("average_ending_money")),
                "epsilon": safe_float(row.get("epsilon")),
            }
        )

    evaluation_series = []
    for row in evaluation_rows:
        episode = safe_int(row.get("episode"))
        if episode is None:
            continue
        evaluation_series.append(
            {
                "episode": episode,
                "average_reward": safe_float(row.get("average_reward")),
                "bankruptcy_rate": safe_float(row.get("bankruptcy_rate")),
                "average_ending_money": safe_float(row.get("average_ending_money")),
                "invalid_action_rate": safe_float(row.get("invalid_action_rate")),
            }
        )

    latest_training = training_series[-1] if training_series else None
    latest_evaluation = evaluation_series[-1] if evaluation_series else None
    latest_episode = latest_training["episode"] if latest_training else 0
    completed_episodes = max(0, latest_episode - start_episode + 1)
    ratio = 0.0
    if total_episodes and total_episodes > 0:
        ratio = max(0.0, min(1.0, completed_episodes / total_episodes))

    progress.update(
        {
            "latest_episode": latest_episode,
            "completed_episodes": completed_episodes,
            "progress_ratio": ratio,
            "latest_training_row": latest_training,
            "latest_evaluation_row": latest_evaluation,
            "training_series": training_series,
            "evaluation_series": evaluation_series,
        }
    )
    return progress


def get_job_log_tail(job_id: int, max_lines: int = 80) -> dict | None:
    init_db()
    refresh_job_states()
    job = get_job(job_id)
    if job is None:
      return None

    log_path_raw = job.get("stdout_log_path") or ""
    log_path = Path(log_path_raw) if log_path_raw else None
    if not log_path or not log_path.exists():
        return {
            "job_id": job_id,
            "stdout_log_path": log_path_raw,
            "lines": [],
        }

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    return {
        "job_id": job_id,
        "stdout_log_path": str(log_path),
        "lines": [line.rstrip("\n") for line in lines[-max_lines:]],
    }


def enqueue_train_job(payload: dict) -> dict:
    init_db()
    resume_from = payload.get("resume_from")
    resume_mode = payload.get("resume_mode", "strict")
    job_type = "fine_tune" if resume_from and resume_mode in {"fine_tune", "fine-tune"} else "train"
    run_dir = create_job_run_dir(job_type, run_name=payload.get("run_name"))
    stdout_log_path = default_stdout_log(run_dir, job_type)

    job = create_job(
        job_type=job_type,
        payload=payload,
        stdout_log_path=str(stdout_log_path),
        run_dir=str(run_dir),
    )
    dispatch_next_job()
    return get_job(job["id"])


def enqueue_evaluation_job(payload: dict) -> dict:
    init_db()
    job_type = payload.get("job_type", "robustness")
    if job_type not in {"evaluate", "robustness"}:
        raise ValueError("Only evaluate and robustness job types are supported.")

    run_dir = Path(payload["run_dir"]).resolve()
    stdout_log_path = default_stdout_log(run_dir, job_type)
    result_path = str(run_dir / "robustness_results.csv")

    job = create_job(
        job_type=job_type,
        payload=payload,
        stdout_log_path=str(stdout_log_path),
        run_dir=str(run_dir),
        result_path=result_path,
    )
    dispatch_next_job()
    return get_job(job["id"])


def enqueue_ruleset_search_job(payload: dict) -> dict:
    """Queue a ruleset search job which explores randomized configs."""
    init_db()
    job_type = "ruleset_search"
    run_dir = create_job_run_dir(job_type, run_name=payload.get("run_name"))
    stdout_log_path = default_stdout_log(run_dir, job_type)

    job = create_job(
        job_type=job_type,
        payload=payload,
        stdout_log_path=str(stdout_log_path),
        run_dir=str(run_dir),
    )
    dispatch_next_job()
    return get_job(job["id"])


def stop_job(job_id: int) -> dict:
    init_db()
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"Job `{job_id}` was not found.")

    if job["status"] == "queued":
        return update_job(job_id, status="stopped", ended_at=utc_now_iso(), error_message="Stopped before execution.")

    if job["status"] != "running":
        return job

    pid = job.get("worker_pid")
    if is_pid_running(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True)
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                os.kill(pid, signal.SIGTERM)

    updated = update_job(job_id, status="stopped", ended_at=utc_now_iso(), error_message="Stopped by user request.")
    dispatch_next_job()
    return updated


def dismiss_job(job_id: int) -> dict:
    init_db()
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"Job `{job_id}` was not found.")

    if job["status"] in {"queued", "running"}:
        raise ValueError(f"Job `{job_id}` cannot be dismissed while it is {job['status']}.")

    deleted = delete_job(job_id)
    if not deleted:
        raise ValueError(f"Job `{job_id}` could not be dismissed.")

    return {"ok": True, "job_id": job_id}
