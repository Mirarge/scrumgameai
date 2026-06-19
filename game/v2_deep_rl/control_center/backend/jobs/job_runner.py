from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.app_paths import BACKEND_DIR, ENGINE_ROOT
from storage.jobs_db import get_job, init_db, update_job, utc_now_iso

_CACHED_PYTHON_COMMAND: str | None = None


def _choose_python_command() -> str:
    """Return a Python executable that can import torch.

    job_runner may itself be spawned by the venv Python (which lacks torch),
    so we must not blindly use sys.executable to run training scripts.
    """
    global _CACHED_PYTHON_COMMAND
    if _CACHED_PYTHON_COMMAND is not None:
        return _CACHED_PYTHON_COMMAND

    try:
        import torch  # noqa: F401
        _CACHED_PYTHON_COMMAND = sys.executable
        return _CACHED_PYTHON_COMMAND
    except ImportError:
        pass

    import shutil
    for candidate in ("python", "python3", "py"):
        exe = shutil.which(candidate)
        if not exe:
            continue
        try:
            probe = subprocess.run(
                [exe, "-c", "import torch"],
                capture_output=True,
                timeout=15,
            )
            if probe.returncode == 0:
                _CACHED_PYTHON_COMMAND = exe
                return _CACHED_PYTHON_COMMAND
        except Exception:
            continue

    _CACHED_PYTHON_COMMAND = sys.executable
    return _CACHED_PYTHON_COMMAND


def build_command(job: dict) -> list[str]:
    payload = job["payload"]
    python_command = _choose_python_command()

    if job["job_type"] in {"train", "fine_tune"}:
        command = [
            python_command,
            "-m",
            "training.train_dqn",
            "--run-dir",
            str(job["run_dir"]),
        ]
        if payload.get("game_config_path"):
            command.extend(["--game-config", str(payload["game_config_path"])])
        if payload.get("training_config_path"):
            command.extend(["--training-config", str(payload["training_config_path"])])
        if payload.get("episodes") is not None:
            command.extend(["--episodes", str(int(payload["episodes"]))])
        if payload.get("evaluation_episodes") is not None:
            command.extend(["--evaluation-episodes", str(int(payload["evaluation_episodes"]))])
        if payload.get("learning_rate") is not None:
            command.extend(["--learning-rate", str(payload["learning_rate"])])
        if payload.get("gamma") is not None:
            command.extend(["--gamma", str(payload["gamma"])])
        if payload.get("epsilon_decay_end") is not None:
            command.extend(["--epsilon-decay-episodes", str(float(payload["epsilon_decay_end"]))])
        if payload.get("rule_randomization_enabled"):
            command.append("--rule-randomization")
        if payload.get("rule_randomization_frequency") is not None:
            command.extend(["--rule-randomization-frequency", str(int(payload["rule_randomization_frequency"]))])
        if payload.get("rule_randomization_eval_configs") is not None:
            command.extend(["--rule-randomization-eval-configs", str(int(payload["rule_randomization_eval_configs"]))])
        if payload.get("rule_randomization_bounds") is not None:
            command.extend(["--rule-randomization-bounds-json", json.dumps(payload["rule_randomization_bounds"])])
        if payload.get("auto_continue_enabled"):
            command.append("--auto-continue")
        if payload.get("auto_continue_cycles") is not None:
            command.extend(["--auto-continue-cycles", str(int(payload["auto_continue_cycles"]))])
        if payload.get("seed") is not None:
            command.extend(["--seed", str(int(payload["seed"]))])
        if payload.get("run_notes"):
            command.extend(["--notes", str(payload["run_notes"])])
        if payload.get("resume_from"):
            command.extend(["--resume-from", str(payload["resume_from"])])
            command.extend(["--resume-mode", str(payload.get("resume_mode", "strict"))])
            if payload.get("resume_episodes_mode"):
                command.extend(["--resume-episodes-mode", str(payload["resume_episodes_mode"])])
        return command

    if job["job_type"] in {"evaluate", "robustness"}:
        return [
            python_command,
            "-m",
            "evaluation.evaluate_ddqn_robustness",
            "--run-dir",
            str(payload["run_dir"]),
        ]

    if job["job_type"] == "ruleset_search":
        command = [
            python_command,
            "-m",
            "evaluation.ruleset_search",
            "--run-dir",
            str(job["run_dir"]),
        ]
        if payload.get("game_config_path"):
            command.extend(["--game-config", str(payload.get("game_config_path"))])
        if payload.get("initial_games") is not None:
            command.extend(["--initial-games", str(int(payload.get("initial_games")))])
        if payload.get("max_rulesets") is not None:
            command.extend(["--max-rulesets", str(int(payload.get("max_rulesets")))])
        if payload.get("promising_target") is not None:
            command.extend(["--promising-target", str(float(payload.get("promising_target")))])
        if payload.get("leniency") is not None:
            command.extend(["--leniency", str(float(payload.get("leniency")))])
        if payload.get("retrain_episodes") is not None:
            command.extend(["--retrain-episodes", str(int(payload.get("retrain_episodes")))])
        if payload.get("max_promising") is not None:
            command.extend(["--max-promising", str(int(payload.get("max_promising")))])
        if payload.get("final_choices") is not None:
            command.extend(["--final-choices", str(int(payload.get("final_choices")))])
        if payload.get("final_train_episodes") is not None:
            command.extend(["--final-train-episodes", str(int(payload.get("final_train_episodes")))])
        if payload.get("bounds_json") is not None:
            command.extend(["--bounds-json", json.dumps(payload.get("bounds_json"))])
        return command

    raise ValueError(f"Unsupported job type: {job['job_type']}")


def run_job(job_id: int) -> int:
    init_db()
    job = get_job(job_id)
    if job is None:
        raise RuntimeError(f"Job `{job_id}` was not found.")

    stdout_log_path = Path(job["stdout_log_path"])
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        command = build_command(job)

        with stdout_log_path.open("ab") as stdout_handle:
            result = subprocess.run(
                command,
                cwd=ENGINE_ROOT,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                check=False,
            )

        status = "completed" if result.returncode == 0 else "failed"
        error_message = None if result.returncode == 0 else f"Command exited with code {result.returncode}."
        update_job(
            job_id,
            status=status,
            ended_at=utc_now_iso(),
            error_message=error_message,
        )
        return_code = result.returncode
    except Exception as error:
        update_job(
            job_id,
            status="failed",
            ended_at=utc_now_iso(),
            error_message=str(error),
        )
        return_code = 1

    # If a training job completed successfully and was started by the autopilot,
    # trigger the next autopilot cycle before dispatching the next queued job.
    # Manual jobs without autopilot_after_completion do NOT trigger autopilot.
    if return_code == 0 and job["job_type"] in {"train", "fine_tune"}:
        payload = job.get("payload") or {}
        auto_continue_from_config = False
        try:
            training_config_path = Path(job["run_dir"]) / "training_config.json"
            if training_config_path.exists():
                training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
                auto_continue_from_config = bool(training_config.get("auto_continue_enabled", False))
        except Exception:
            auto_continue_from_config = False

        if payload.get("autopilot_after_completion") or auto_continue_from_config:
            run_id = Path(job["run_dir"]).name
            context = payload.get("autopilot_context") or {}
            try:
                from services.training_autopilot import get_settings, run_autopilot
                if get_settings().get("logic_enabled", True):
                    run_autopilot(run_id, context=context)
            except Exception as autopilot_error:
                with stdout_log_path.open("ab") as log_handle:
                    log_handle.write(f"\n[autopilot] Error during autopilot cycle: {autopilot_error}\n".encode())
        else:
            # Manual (non-autopilot) training job — auto-queue a robustness evaluation
            # so the user gets immediate feedback without having to trigger it manually.
            best_pth = Path(job["run_dir"]) / "checkpoints" / "best_scrum_model.pth"
            if best_pth.exists():
                try:
                    from jobs.queue_manager import enqueue_evaluation_job
                    enqueue_evaluation_job({
                        "job_type": "robustness",
                        "run_dir": str(Path(job["run_dir"]).resolve()),
                    })
                except Exception as eval_error:
                    with stdout_log_path.open("ab") as log_handle:
                        log_handle.write(f"\n[auto-eval] Could not queue robustness job: {eval_error}\n".encode())

    # Hand off immediately to the next queued job once this worker has
    # persisted its terminal state.
    from jobs.queue_manager import dispatch_next_job

    dispatch_next_job()
    return return_code


def parse_args():
    parser = argparse.ArgumentParser(description="Run one queued Control Center job.")
    parser.add_argument("--job-id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_job(args.job_id))
