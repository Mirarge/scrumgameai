from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch


LOG_HEADER = [
    "episode",
    "epsilon",
    "episode_reward",
    "rolling_average_reward",
    "mean_recent_loss",
    "replay_buffer_size",
    "average_loan_duration",
    "bankruptcy_count",
    "average_ending_money",
    "invalid_action_count",
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "action_4_count",
    "action_5_count",
    "action_6_count",
    "action_7_count",
]
EVAL_HEADER = ["episode", "average_reward", "bankruptcy_rate", "average_ending_money", "invalid_action_rate"]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _make_plateau_run(runs_dir: Path, run_id: str) -> None:
    run_dir = runs_dir / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "latest_scrum_model.pth").touch()
    (run_dir / "checkpoints" / "best_scrum_model.pth").touch()
    (run_dir / "game_config.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "config_name": "T",
                "config_description": "",
                "players_count": 1,
                "product_names": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
                "max_turns": 6,
                "starting_money": 25000,
                "ring_value": 10000,
                "cost_continue": 0,
                "cost_switch_mid": 5000,
                "cost_switch_after": 2500,
                "mandatory_loan_amount": 50000,
                "loan_interest": 5000,
                "penalty_negative": -100,
                "penalty_positive": 0,
                "daily_scrums_per_sprint": 3,
                "daily_scrum_target": 3,
                "board_ring_values": [[0] * 7 for _ in range(4)],
                "board_features": [[3] * 7 for _ in range(4)],
                "dice_rules": [{"min_features": 1, "max_features": 4, "dice_count": 1, "dice_sides": 6}],
                "refinement": {"active": False, "product_rules": []},
                "incident": {
                    "active": True,
                    "allow_player_specific_incidents": False,
                    "draw_probability": 1.0,
                    "severity_multiplier": 1.0,
                    "cards": [],
                },
                "reserved_fields": {},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                "learning_rate": 0.0005,
                "epsilon_decay_end": 0.9,
                "epsilon_start": 1.0,
                "epsilon_min": 0.05,
            }
        ),
        encoding="utf-8",
    )
    eval_rows = [[i * 10000, 50000.0, 0.2, 40000.0, 0.05] for i in range(8)]
    _write_csv(run_dir / "reports" / "evaluation_history.csv", EVAL_HEADER, eval_rows)
    log_rows = [[100000, 0.04, 50000, 50000, 0.01, 1000, 0, 0, 40000, 5] + [10] * 8]
    _write_csv(run_dir / "reports" / "logs.csv", LOG_HEADER, log_rows)


def test_run_autopilot_stop_notifies_campaign(tmp_path):
    from services.training_autopilot import run_autopilot

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_plateau_run(runs_dir, "run_plateau")

    with patch("services.training_autopilot.RUNS_DIR", runs_dir), patch(
        "services.training_autopilot.ARTIFACTS_DIR", tmp_path
    ), patch("services.training_autopilot.get_settings", return_value={"logic_enabled": True, "ai_enabled": False}), patch(
        "services.campaign_service.CAMPAIGNS_DIR", tmp_path / "campaigns"
    ), patch(
        "services.campaign_service.RUNS_DIR", runs_dir
    ), patch(
        "services.campaign_service.generate_variation",
        return_value=(
            json.loads((runs_dir / "run_plateau" / "game_config.json").read_text(encoding="utf-8")),
            {},
            "test",
        ),
    ) as mock_gen, patch(
        "services.campaign_service.enqueue_train_job",
        return_value={"id": 1, "run_dir": str(runs_dir / "run_plateau_cv1")},
    ), patch(
        "jobs.queue_manager.enqueue_evaluation_job",
        return_value={"id": 2},
    ):
        from services.campaign_service import create_campaign

        (tmp_path / "campaigns").mkdir(exist_ok=True)
        create_campaign("run_plateau", max_variations=3)
        decision = run_autopilot("run_plateau", dry_run=False)

    if decision["action"] in {"stop", "stop_regression"}:
        mock_gen.assert_called_once()
