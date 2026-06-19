"""
Tests for the autopilot continuation chain.

Regression coverage for the bug where every continuation run replayed the same
episodes because analyze_run used best_scrum_model.pth (saved at resume_start)
instead of latest_scrum_model.pth (saved at final_episode).
"""
import csv
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EVAL_HEADER = ["episode", "average_reward", "bankruptcy_rate", "average_ending_money", "invalid_action_rate"]
LOG_HEADER = [
    "episode", "epsilon", "episode_reward", "rolling_average_reward",
    "mean_recent_loss", "replay_buffer_size", "average_loan_duration",
    "bankruptcy_count", "average_ending_money", "invalid_action_count",
    "action_0_count", "action_1_count", "action_2_count", "action_3_count",
    "action_4_count", "action_5_count", "action_6_count", "action_7_count",
]


def _write_csv(path: Path, header: list, rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_run(
    runs_dir: Path,
    run_id: str,
    *,
    final_epsilon: float = 0.789,
    final_episode: int = 100_000,
    eval_rewards: list[float] | None = None,
    has_latest_checkpoint: bool = True,
    has_best_checkpoint: bool = True,
    latest_episode: int | None = None,
    best_episode: int | None = None,
) -> Path:
    """Create a minimal fake run directory for autopilot tests."""
    run_dir = runs_dir / run_id
    reports = run_dir / "reports"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    # training_config.json
    (run_dir / "training_config.json").write_text(json.dumps({
        "learning_rate": 0.0005,
        "epsilon_decay_end": 0.9,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
    }))

    # logs.csv — single row at final_episode with given epsilon
    _write_csv(
        reports / "logs.csv",
        LOG_HEADER,
        [[final_episode, final_epsilon] + [0] * (len(LOG_HEADER) - 2)],
    )

    # evaluation_history.csv
    rewards = eval_rewards or [-5000.0, -4500.0, -4000.0, -3500.0, -3000.0]
    eval_rows = [[
        (i + 1) * 10_000,  # episode
        r,                  # average_reward
        0.3,                # bankruptcy_rate
        10_000.0,           # average_ending_money
        0.05,               # invalid_action_rate
    ] for i, r in enumerate(rewards)]
    _write_csv(reports / "evaluation_history.csv", EVAL_HEADER, eval_rows)

    # Checkpoint stubs (analyze_run only checks existence, never loads them)
    if has_latest_checkpoint:
        p = checkpoints / "latest_scrum_model.pth"
        p.write_bytes(b"stub")
        meta = {"episode": latest_episode or final_episode, "average_reward": -3000.0}
        p.with_suffix(".json").write_text(json.dumps(meta))

    if has_best_checkpoint:
        p = checkpoints / "best_scrum_model.pth"
        p.write_bytes(b"stub")
        meta = {"episode": best_episode or (final_episode - 50_000), "average_reward": -3000.0}
        p.with_suffix(".json").write_text(json.dumps(meta))

    return run_dir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    """Redirect RUNS_DIR inside training_autopilot to a temp directory."""
    import services.training_autopilot as autopilot
    monkeypatch.setattr(autopilot, "RUNS_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: checkpoint path selection
# ---------------------------------------------------------------------------

class TestCheckpointSelection:
    def test_prefers_latest_over_best(self, runs_dir):
        """analyze_run must use latest_scrum_model.pth when it exists."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_test",
            final_epsilon=0.789,
            final_episode=100_000,
            latest_episode=100_000,
            best_episode=50_000,  # would cause chain replay if used
        )

        decision = autopilot.analyze_run("run_test")
        resume_from = decision["next_payload"]["resume_from"]
        assert "latest_scrum_model.pth" in resume_from, (
            "autopilot must resume from latest_scrum_model.pth, not best_scrum_model.pth"
        )

    def test_falls_back_to_best_when_no_latest(self, runs_dir):
        """Without latest_scrum_model.pth, analyze_run falls back to best."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_test",
            final_epsilon=0.789,
            has_latest_checkpoint=False,
            has_best_checkpoint=True,
        )

        decision = autopilot.analyze_run("run_test")
        resume_from = decision["next_payload"]["resume_from"]
        assert "best_scrum_model.pth" in resume_from

    def test_no_checkpoint_returns_none(self, runs_dir):
        """Without any checkpoint the resume_from path is None."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_test",
            final_epsilon=0.789,
            has_latest_checkpoint=False,
            has_best_checkpoint=False,
        )

        decision = autopilot.analyze_run("run_test")
        assert decision["next_payload"]["resume_from"] is None


# ---------------------------------------------------------------------------
# Tests: chain replay regression
# ---------------------------------------------------------------------------

class TestChainReplayRegression:
    def test_latest_episode_advances_on_each_continuation(self, runs_dir):
        """
        Regression: if analyze_run uses best_scrum_model.pth (episode 50k) instead
        of latest_scrum_model.pth (episode 100k), every continuation re-runs
        episodes 50k-100k identically.

        This test verifies that the resume_from checkpoint has the FINAL episode,
        not the best-reward episode from earlier in the run.
        """
        import json as _json
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_v2",
            final_epsilon=0.789,
            final_episode=100_000,
            latest_episode=100_000,   # end of run
            best_episode=50_000,      # reward peaked early — old bug used this
        )

        decision = autopilot.analyze_run("run_v2")
        resume_from = decision["next_payload"]["resume_from"]

        # Load the sidecar of the chosen checkpoint and verify its episode
        sidecar = Path(resume_from).with_suffix(".json")
        meta = _json.loads(sidecar.read_text())
        assert meta["episode"] == 100_000, (
            f"Continuation must start from episode 100k, got {meta['episode']}. "
            "This would cause the chain to replay the same episodes."
        )


# ---------------------------------------------------------------------------
# Tests: epsilon exploration gate
# ---------------------------------------------------------------------------

class TestEpsilonGate:
    def test_high_epsilon_skips_plateau_check(self, runs_dir):
        """Epsilon > 0.5 → action=continue, no plateau analysis."""
        import services.training_autopilot as autopilot

        _make_run(runs_dir, "run_test", final_epsilon=0.789)

        decision = autopilot.analyze_run("run_test")
        assert decision["action"] == "continue"
        assert "exploration phase" in decision["reason"]
        assert decision["metrics"]["reward_improvement_ratio"] is None

    def test_low_epsilon_enables_plateau_check(self, runs_dir):
        """Epsilon < 0.5 → plateau logic runs."""
        import services.training_autopilot as autopilot

        # Flat rewards → should detect plateau
        _make_run(
            runs_dir, "run_test",
            final_epsilon=0.30,
            eval_rewards=[-5000.0, -5010.0, -4990.0, -5005.0],
        )

        decision = autopilot.analyze_run("run_test")
        assert decision["metrics"]["reward_improvement_ratio"] is not None

    @pytest.mark.parametrize("epsilon", [0.51, 0.789, 0.894, 1.0])
    def test_various_high_epsilons_all_continue(self, runs_dir, epsilon):
        import services.training_autopilot as autopilot
        _make_run(runs_dir, "run_test", final_epsilon=epsilon)
        decision = autopilot.analyze_run("run_test")
        assert decision["action"] == "continue"

    @pytest.mark.parametrize("epsilon", [0.0, 0.25, 0.49])
    def test_various_low_epsilons_run_plateau_check(self, runs_dir, epsilon):
        import services.training_autopilot as autopilot
        _make_run(runs_dir, "run_test", final_epsilon=epsilon,
                  eval_rewards=[-5000.0, -5010.0, -4990.0, -5005.0])
        decision = autopilot.analyze_run("run_test")
        assert decision["metrics"]["reward_improvement_ratio"] is not None


# ---------------------------------------------------------------------------
# Tests: continuation payload
# ---------------------------------------------------------------------------

class TestContinuationPayload:
    def test_continue_uses_strict_resume_mode(self, runs_dir):
        """action=continue must use strict resume to carry over optimizer + replay."""
        import services.training_autopilot as autopilot
        _make_run(runs_dir, "run_test", final_epsilon=0.789)

        decision = autopilot.analyze_run("run_test")
        assert decision["next_payload"]["resume_mode"] == "strict"

    def test_continue_uses_incremental_episodes(self, runs_dir):
        import services.training_autopilot as autopilot
        _make_run(runs_dir, "run_test", final_epsilon=0.789)

        decision = autopilot.analyze_run("run_test")
        assert decision["next_payload"]["resume_episodes_mode"] == "incremental"

    def test_continue_episodes_equals_constant(self, runs_dir):
        import services.training_autopilot as autopilot
        _make_run(runs_dir, "run_test", final_epsilon=0.789)

        decision = autopilot.analyze_run("run_test")
        assert decision["next_payload"]["episodes"] == autopilot.CONTINUE_EPISODES
