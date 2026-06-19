from __future__ import annotations

from fastapi import APIRouter, HTTPException

from jobs.queue_manager import (
    dismiss_job,
    enqueue_evaluation_job,
    enqueue_train_job,
    get_job_details,
    enqueue_ruleset_search_job,
    get_job_log_tail,
    get_job_progress,
    list_jobs,
    stop_job,
)


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def get_jobs():
    """List queued, running, completed, failed, and stopped jobs."""
    return {"items": list_jobs()}


@router.get("/{job_id}/progress")
def get_job_progress_route(job_id: int):
    """Return live progress and chart data for one job when available."""
    job = get_job_progress(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job `{job_id}` was not found.")
    return job


@router.get("/{job_id}/log")
def get_job_log_route(job_id: int, max_lines: int = 80):
    """Return the tail of one job stdout log."""
    payload = get_job_log_tail(job_id, max_lines=max_lines)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Job `{job_id}` was not found.")
    return payload


@router.get("/{job_id}")
def get_job(job_id: int):
    """Return one queued job and its persisted state."""
    job = get_job_details(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job `{job_id}` was not found.")
    return job


@router.post("/train")
def create_training_job(payload: dict):
    """Queue a training, resume, or fine-tune job."""
    try:
        return enqueue_train_job(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@router.post("/ruleset-search")
@router.post("/ruleset_search")
def create_ruleset_search_job(payload: dict):
    """Queue a ruleset search job to find promising randomized configs."""
    try:
        return enqueue_ruleset_search_job(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/evaluate")
def create_evaluation_job(payload: dict):
    """Queue an evaluation or robustness job against an existing run directory."""
    try:
        return enqueue_evaluation_job(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/{job_id}/stop")
def stop_job_route(job_id: int):
    """Stop a queued or running job."""
    try:
        return stop_job(job_id)
    except ValueError as error:
        detail = str(error)
        status_code = 404 if "was not found" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from error


@router.delete("/{job_id}")
def dismiss_job_route(job_id: int):
    """Dismiss one terminal job from the queue list."""
    try:
        return dismiss_job(job_id)
    except ValueError as error:
        detail = str(error)
        status_code = 404 if "was not found" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from error
