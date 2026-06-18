from __future__ import annotations

from services.app_paths import RUNS_DIR
from services.io_utils import read_json_safe, safe_float, safe_int, tail_csv_rows


def list_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for run_dir in sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
        metadata_path = run_dir / "run_metadata.json"
        metrics_path = run_dir / "reports" / "dqn_metrics.json"
        checkpoint_path = run_dir / "checkpoints" / "best_scrum_model.pth"
        metadata = read_json_safe(metadata_path)
        metrics = read_json_safe(metrics_path)

        runs.append(
            {
                "id": run_dir.name,
                "label": run_dir.name,
                "path": str(run_dir),
                "created_at": metadata.get("created_at"),
                "run_notes": metadata.get("run_notes", ""),
                "rule_signature": metadata.get("rule_signature"),
                "training_signature": metadata.get("training_signature"),
                "resume_mode": metadata.get("resume_mode"),
                "resume_checkpoint_path": metadata.get("resume_checkpoint_path"),
                "best_checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
                "metrics_path": str(metrics_path) if metrics_path.exists() else None,
                "average_reward_per_episode": metrics.get("average_reward_per_episode"),
                "bankruptcy_rate": metrics.get("bankruptcy_rate"),
            }
        )
    return runs


def get_run(run_id: str) -> dict | None:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    metadata_path = run_dir / "run_metadata.json"
    metrics_path = run_dir / "reports" / "dqn_metrics.json"
    game_config_path = run_dir / "game_config.json"
    training_config_path = run_dir / "training_config.json"

    return {
        "id": run_dir.name,
        "label": run_dir.name,
        "path": str(run_dir),
        "metadata": read_json_safe(metadata_path),
        "metrics": read_json_safe(metrics_path),
        "game_config": read_json_safe(game_config_path) if game_config_path.exists() else None,
        "training_config": read_json_safe(training_config_path) if training_config_path.exists() else None,
        "checkpoints": [
            {
                "name": checkpoint_path.name,
                "path": str(checkpoint_path),
            }
            for checkpoint_path in sorted((run_dir / "checkpoints").glob("*.pth"))
        ],
    }


def get_run_progress(run_id: str) -> dict | None:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    metadata_path = run_dir / "run_metadata.json"
    run_metadata = read_json_safe(metadata_path)
    training_config_path = run_dir / "training_config.json"
    training_config = read_json_safe(training_config_path)
    total_episodes = safe_int(str(run_metadata.get("episodes_this_run", "")))
    if total_episodes is None:
        total_episodes = safe_int(str(training_config.get("episodes", "")))
    start_episode = safe_int(str(run_metadata.get("start_episode", ""))) or 1

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

    return {
        "run_id": run_id,
        "job_id": None,
        "job_type": "train",
        "status": "completed",
        "run_dir": str(run_dir),
        "stdout_log_path": "",
        "error_message": None,
        "total_episodes": total_episodes,
        "start_episode": start_episode,
        "latest_episode": latest_episode,
        "completed_episodes": completed_episodes,
        "progress_ratio": ratio,
        "latest_training_row": latest_training,
        "latest_evaluation_row": latest_evaluation,
        "training_series": training_series,
        "evaluation_series": evaluation_series,
    }
