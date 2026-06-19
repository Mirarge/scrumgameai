"""
Tests for analyze_run decision branches in training_autopilot.py that are NOT
already covered by test_autopilot_continuation.py:

  - stop branch:          flat rewards, low invalid action rate
  - lower_lr branch:      improving rewards but high variance (CV > 0.20)
  - extend_epsilon_decay: flat rewards + high invalid action rate (> 0.10)
  - lr_reduction_count cap: after MAX_LR_REDUCTIONS → action=stop
  - AI advisor skipped during dry_run
  - run_autopilot versioned naming: base_run_name derived + continuation_version increments

The _make_run helper is duplicated here (rather than extracted to conftest) so
this test file remains fully self-contained and doesn't pollute fixtures shared
with other test files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test-local run-builder helpers  (mirrors pattern from test_autopilot_continuation.py)
# ---------------------------------------------------------------------------

EVAL_HEADER = [
    "episode", "average_reward", "bankruptcy_rate",
    "average_ending_money", "invalid_action_rate",
]
LOG_HEADER = [
    "episode", "epsilon", "episode_reward", "rolling_average_reward",
    "mean_recent_loss", "replay_buffer_size", "average_loan_duration",
    "bankruptcy_count", "average_ending_money", "invalid_action_count",
    "action_0_count", "action_1_count", "action_2_count", "action_3_count",
    "action_4_count", "action_5_count", "action_6_count", "action_7_count",
]


def _write_csv(path: Path, header: list, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_run(
    runs_dir: Path,
    run_id: str,
    *,
    final_epsilon: float = 0.10,
    final_episode: int = 100_000,
    eval_rewards: list[float] | None = None,
    invalid_action_rate: float = 0.05,
    bankruptcy_rate: float = 0.30,
    bankruptcy_rates: list[float] | None = None,
    has_latest_checkpoint: bool = True,
    has_best_checkpoint: bool = True,
    learning_rate: float = 0.0005,
    epsilon_decay_end: float = 0.9,
) -> Path:
    """Create a minimal fake run directory for autopilot decision tests."""
    run_dir = runs_dir / run_id
    reports = run_dir / "reports"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    # training_config.json
    (run_dir / "training_config.json").write_text(json.dumps({
        "learning_rate": learning_rate,
        "epsilon_decay_end": epsilon_decay_end,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
    }))

    # logs.csv — one row at final_episode carrying the supplied epsilon
    _write_csv(
        reports / "logs.csv",
        LOG_HEADER,
        [[final_episode, final_epsilon] + [0] * (len(LOG_HEADER) - 2)],
    )

    # evaluation_history.csv — 4 rows (== PLATEAU_WINDOW) by default
    rewards = eval_rewards or [-5000.0, -5010.0, -4990.0, -5005.0]
    br_list = bankruptcy_rates if bankruptcy_rates and len(bankruptcy_rates) == len(rewards) else [bankruptcy_rate] * len(rewards)
    eval_rows = [
        [
            (i + 1) * 10_000,   # episode
            r,                   # average_reward
            br,                  # bankruptcy_rate
            10_000.0,            # average_ending_money
            invalid_action_rate, # invalid_action_rate
        ]
        for i, (r, br) in enumerate(zip(rewards, br_list))
    ]
    _write_csv(reports / "evaluation_history.csv", EVAL_HEADER, eval_rows)

    # Checkpoint stubs
    if has_latest_checkpoint:
        p = checkpoints / "latest_scrum_model.pth"
        p.write_bytes(b"stub")
        p.with_suffix(".json").write_text(json.dumps({"episode": final_episode}))

    if has_best_checkpoint:
        p = checkpoints / "best_scrum_model.pth"
        p.write_bytes(b"stub")
        p.with_suffix(".json").write_text(json.dumps({"episode": final_episode - 50_000}))

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
# STOP branch
# ---------------------------------------------------------------------------

class TestStopBranch:
    def test_flat_rewards_low_invalid_rate_stops(self, runs_dir):
        """Flat reward window + low invalid action rate must produce action=stop."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_stop",
            final_epsilon=0.10,
            # Essentially flat: 0.1% improvement over window (below 2% threshold)
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.03,  # below INVALID_ACTION_HIGH (0.10)
        )

        decision = autopilot.analyze_run("run_stop")
        assert decision["action"] == "stop"

    def test_stop_next_payload_is_none(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_stop",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_stop")
        assert decision["next_payload"] is None

    def test_stop_reason_mentions_plateau(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_stop",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5005.0, -4998.0, -5003.0],
            invalid_action_rate=0.02,
        )

        decision = autopilot.analyze_run("run_stop")
        assert "plateau" in decision["reason"].lower()

    def test_stop_includes_bankruptcy_rate_in_reason(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_stop",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5005.0, -4998.0, -5003.0],
            invalid_action_rate=0.02,
            bankruptcy_rate=0.45,
        )

        decision = autopilot.analyze_run("run_stop")
        assert "45" in decision["reason"] or "bankruptcy" in decision["reason"].lower()

    def test_stop_improvement_metric_near_zero(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_stop",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5000.0, -5000.0, -5000.0],
            invalid_action_rate=0.0,
        )

        decision = autopilot.analyze_run("run_stop")
        assert decision["action"] == "stop"
        assert decision["metrics"]["reward_improvement_ratio"] == pytest.approx(0.0)

    @pytest.mark.parametrize("rewards", [
        [-100.0, -101.0, -99.5, -100.5],   # tiny negative swing
        [-200.0, -201.0, -200.5, -200.8],  # near-flat positive numbers
        [-1.0, -1.01, -0.99, -1.005],      # small values
    ])
    def test_various_flat_reward_shapes_stop(self, runs_dir, rewards):
        """Any flat-reward window must resolve to stop when invalid_action_rate is low."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_stop",
            final_epsilon=0.10,
            eval_rewards=rewards,
            invalid_action_rate=0.01,
        )

        decision = autopilot.analyze_run("run_stop")
        assert decision["action"] == "stop"


# ---------------------------------------------------------------------------
# lower_lr branch
# ---------------------------------------------------------------------------

class TestLowerLrBranch:
    def _high_variance_rewards(self) -> list[float]:
        """Four rewards that improve > 2% overall but have high CV (> 0.20)."""
        # Mean ≈ -3875, std large due to big swings, improvement = (-1000 - -5000) / 5000 = 80%
        return [-5000.0, -2000.0, -6000.0, -1000.0]

    def test_improving_high_variance_gives_lower_lr(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_lr",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
            invalid_action_rate=0.05,
        )

        decision = autopilot.analyze_run("run_lr")
        assert decision["action"] == "lower_lr"

    def test_lower_lr_halves_learning_rate(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_lr",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
            learning_rate=0.001,
        )

        decision = autopilot.analyze_run("run_lr")
        assert decision["action"] == "lower_lr"
        new_lr = decision["next_payload"]["learning_rate"]
        assert new_lr == pytest.approx(0.0005, rel=1e-4)

    def test_lower_lr_uses_fine_tune_resume_mode(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_lr",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run("run_lr")
        assert decision["action"] == "lower_lr"
        assert decision["next_payload"]["resume_mode"] == "fine-tune"

    def test_lower_lr_payload_has_continue_episodes(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_lr",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run("run_lr")
        assert decision["next_payload"]["episodes"] == autopilot.CONTINUE_EPISODES

    def test_lower_lr_does_not_go_below_lr_floor(self, runs_dir):
        """LR halving must be clamped to _LR_MIN."""
        import services.training_autopilot as autopilot

        # Start at the minimum already — should stay at _LR_MIN
        _make_run(
            runs_dir, "run_lr",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
            learning_rate=autopilot._LR_MIN,
        )

        decision = autopilot.analyze_run("run_lr")
        if decision["action"] == "lower_lr":
            new_lr = decision["next_payload"]["learning_rate"]
            assert new_lr >= autopilot._LR_MIN

    def test_lower_lr_reason_mentions_noise(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_lr",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run("run_lr")
        assert decision["action"] == "lower_lr"
        reason_lower = decision["reason"].lower()
        assert "noisy" in reason_lower or "variance" in reason_lower or "cv" in reason_lower


# ---------------------------------------------------------------------------
# extend_epsilon_decay branch
# ---------------------------------------------------------------------------

class TestExtendEpsilonDecayBranch:
    def test_flat_rewards_high_invalid_rate_extends_epsilon(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_ext",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.15,   # above INVALID_ACTION_HIGH (0.10)
        )

        decision = autopilot.analyze_run("run_ext")
        assert decision["action"] == "extend_epsilon_decay"

    def test_extend_increases_epsilon_decay_end(self, runs_dir):
        import services.training_autopilot as autopilot

        original_decay = 450_000
        _make_run(
            runs_dir, "run_ext",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.15,
            epsilon_decay_end=original_decay,
        )

        decision = autopilot.analyze_run("run_ext")
        assert decision["action"] == "extend_epsilon_decay"
        new_decay = decision["next_payload"]["epsilon_decay_end"]
        assert new_decay > original_decay

    def test_extend_uses_extension_factor(self, runs_dir):
        """New decay = original + original * (EPSILON_EXTENSION_FACTOR - 1)."""
        import services.training_autopilot as autopilot

        original_decay = 400_000
        _make_run(
            runs_dir, "run_ext",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.20,
            epsilon_decay_end=original_decay,
        )

        decision = autopilot.analyze_run("run_ext")
        assert decision["action"] == "extend_epsilon_decay"

        expected = original_decay + int(original_decay * (autopilot.EPSILON_EXTENSION_FACTOR - 1.0))
        assert decision["next_payload"]["epsilon_decay_end"] == expected

    def test_extend_uses_fine_tune_resume_mode(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_ext",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.15,
        )

        decision = autopilot.analyze_run("run_ext")
        assert decision["action"] == "extend_epsilon_decay"
        assert decision["next_payload"]["resume_mode"] == "fine-tune"

    def test_extend_reason_mentions_invalid_actions(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_ext",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=0.20,
        )

        decision = autopilot.analyze_run("run_ext")
        assert decision["action"] == "extend_epsilon_decay"
        assert "invalid action" in decision["reason"].lower() or "exploration" in decision["reason"].lower()

    def test_boundary_exactly_at_threshold_does_not_extend(self, runs_dir):
        """Invalid rate == INVALID_ACTION_HIGH (0.10) is NOT strictly greater → stop."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_ext_boundary",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=autopilot.INVALID_ACTION_HIGH,  # exactly 0.10
        )

        decision = autopilot.analyze_run("run_ext_boundary")
        # 0.10 is NOT > 0.10, so the else branch fires → stop
        assert decision["action"] == "stop"

    def test_just_above_threshold_extends(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_ext_above",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            invalid_action_rate=autopilot.INVALID_ACTION_HIGH + 0.001,
        )

        decision = autopilot.analyze_run("run_ext_above")
        assert decision["action"] == "extend_epsilon_decay"


# ---------------------------------------------------------------------------
# lr_reduction_count cap → stop
# ---------------------------------------------------------------------------

class TestLrReductionCountCap:
    def _high_variance_rewards(self) -> list[float]:
        return [-5000.0, -2000.0, -6000.0, -1000.0]

    def test_at_max_reductions_stops_instead_of_lower_lr(self, runs_dir):
        """Once lr_reduction_count >= MAX_LR_REDUCTIONS, action must be stop."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_cap",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run(
            "run_cap",
            context={"lr_reduction_count": autopilot.MAX_LR_REDUCTIONS},
        )
        assert decision["action"] == "stop"

    def test_below_max_reductions_still_lower_lr(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_cap",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run(
            "run_cap",
            context={"lr_reduction_count": autopilot.MAX_LR_REDUCTIONS - 1},
        )
        assert decision["action"] == "lower_lr"

    def test_cap_reason_mentions_lr_reductions(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_cap",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run(
            "run_cap",
            context={"lr_reduction_count": autopilot.MAX_LR_REDUCTIONS},
        )
        assert decision["action"] == "stop"
        reason_lower = decision["reason"].lower()
        assert "lr" in reason_lower or "learning rate" in reason_lower or "reduct" in reason_lower

    def test_lr_reduction_count_zero_is_first_reduction(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_cap",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run(
            "run_cap",
            context={"lr_reduction_count": 0},
        )
        assert decision["action"] == "lower_lr"
        assert "1/" in decision["reason"] or "reduction 1" in decision["reason"].lower()

    @pytest.mark.parametrize("count", [3, 5, 10])
    def test_various_counts_above_max_all_stop(self, runs_dir, count):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_cap",
            final_epsilon=0.10,
            eval_rewards=self._high_variance_rewards(),
        )

        decision = autopilot.analyze_run(
            "run_cap",
            context={"lr_reduction_count": count},
        )
        assert decision["action"] == "stop"


# ---------------------------------------------------------------------------
# AI advisor not called during dry_run
# ---------------------------------------------------------------------------

class TestAiAdvisorSkippedOnDryRun:
    def test_dry_run_does_not_call_ai_advisor(self, runs_dir, monkeypatch):
        """run_autopilot with dry_run=True must never invoke _call_ai_advisor."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_dry",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5005.0, -4998.0, -5003.0],
            invalid_action_rate=0.02,  # → stop branch
        )

        ai_mock = MagicMock(return_value={"action": "stop", "reason": "mock", "next_payload": None, "advisor": "ai"})
        monkeypatch.setattr(autopilot, "_call_ai_advisor", ai_mock)

        decision = autopilot.run_autopilot("run_dry", dry_run=True)
        ai_mock.assert_not_called()

    def test_dry_run_writes_decision_record(self, runs_dir):
        """dry_run must still write the autopilot_decisions.jsonl record."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_dry",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5005.0, -4998.0, -5003.0],
            invalid_action_rate=0.02,
        )

        autopilot.run_autopilot("run_dry", dry_run=True)

        record_path = runs_dir / "run_dry" / "reports" / "autopilot_decisions.jsonl"
        assert record_path.exists()
        lines = [l for l in record_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "action" in record

    def test_dry_run_does_not_enqueue_job(self, runs_dir, monkeypatch):
        """dry_run must never call enqueue_train_job regardless of action."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_dry",
            final_epsilon=0.789,   # high epsilon → continue action
        )

        enqueue_mock = MagicMock()
        monkeypatch.setattr("jobs.queue_manager.enqueue_train_job", enqueue_mock, raising=False)

        decision = autopilot.run_autopilot("run_dry", dry_run=True)
        enqueue_mock.assert_not_called()
        assert decision["job_enqueued"] is False

    def test_dry_run_stop_sets_job_enqueued_false(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_dry",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5005.0, -4998.0, -5003.0],
            invalid_action_rate=0.02,
        )

        decision = autopilot.run_autopilot("run_dry", dry_run=True)
        assert decision["job_enqueued"] is False


# ---------------------------------------------------------------------------
# AI advisor disabled (ai_enabled=False)
# ---------------------------------------------------------------------------

class TestAiAdvisorDisabledSetting:
    def test_ai_disabled_does_not_call_advisor_on_stop(self, runs_dir, monkeypatch):
        """When ai_enabled=False, the AI advisor must not be called even if logic says stop."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_no_ai",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5005.0, -4998.0, -5003.0],
            invalid_action_rate=0.02,  # → stop branch
        )

        # Disable AI and prevent stop-flag side-effects
        monkeypatch.setattr(autopilot, "get_settings", lambda: {"logic_enabled": True, "ai_enabled": False})
        monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)

        ai_mock = MagicMock()
        monkeypatch.setattr(autopilot, "_call_ai_advisor", ai_mock)

        # action=stop → enqueue_train_job is never reached, so no DB patch needed.
        decision = autopilot.run_autopilot("run_no_ai", dry_run=False)

        ai_mock.assert_not_called()
        # Logic stop remains stop — AI had no chance to override
        assert decision["action"] == "stop"


# ---------------------------------------------------------------------------
# run_autopilot versioned naming
# ---------------------------------------------------------------------------

class TestRunAutopilotVersionedNaming:
    def _make_improving_run(self, runs_dir, run_id, *, learning_rate=0.0005):
        """Create a run whose epsilon is high → action=continue → job enqueued."""
        _make_run(
            runs_dir,
            run_id,
            final_epsilon=0.789,  # high epsilon → continue (no plateau analysis)
            learning_rate=learning_rate,
        )

    def test_base_run_name_extracted_from_run_id(self, runs_dir, monkeypatch):
        """base_run_name strips the timestamp prefix from the run_id."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200_myexperiment")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            decision = autopilot.run_autopilot("run_2024-01-15_1200_myexperiment")

        call_payload = enqueue_mock.call_args[0][0]
        assert call_payload["autopilot_context"]["base_run_name"] == "myexperiment"

    def test_versioned_run_name_appends_v2_on_first_continuation(self, runs_dir, monkeypatch):
        """First continuation run gets the _v2 suffix."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200_myexperiment")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            autopilot.run_autopilot("run_2024-01-15_1200_myexperiment")

        call_payload = enqueue_mock.call_args[0][0]
        assert call_payload["run_name"] == "myexperiment_v2"

    def test_continuation_version_increments_in_context(self, runs_dir, monkeypatch):
        """The context carried forward must have continuation_version bumped by 1."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200_myexperiment")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            autopilot.run_autopilot("run_2024-01-15_1200_myexperiment")

        call_payload = enqueue_mock.call_args[0][0]
        assert call_payload["autopilot_context"]["continuation_version"] == 3

    def test_run_id_without_name_uses_v_prefix_only(self, runs_dir, monkeypatch):
        """A run_id with no user name component should get 'v2' as the run_name."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            autopilot.run_autopilot("run_2024-01-15_1200")

        call_payload = enqueue_mock.call_args[0][0]
        assert call_payload["run_name"] == "v2"

    def test_base_run_name_strips_v_suffix(self, runs_dir, monkeypatch):
        """When the run_id already has a _vN suffix it must be stripped so the chain doesn't accumulate _v2_v3."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200_myexperiment_v2")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            autopilot.run_autopilot("run_2024-01-15_1200_myexperiment_v2")

        call_payload = enqueue_mock.call_args[0][0]
        base = call_payload["autopilot_context"]["base_run_name"]
        assert not base.endswith("_v2"), (
            f"base_run_name '{base}' must not carry the _vN suffix"
        )
        assert base == "myexperiment"

    def test_context_base_run_name_overrides_derived(self, runs_dir, monkeypatch):
        """If base_run_name is already in context it should be used as-is."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200_shouldbeignored")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            autopilot.run_autopilot(
                "run_2024-01-15_1200_shouldbeignored",
                context={"base_run_name": "override_name", "continuation_version": 4},
            )

        call_payload = enqueue_mock.call_args[0][0]
        assert call_payload["run_name"] == "override_name_v4"
        assert call_payload["autopilot_context"]["base_run_name"] == "override_name"

    def test_continuation_version_from_context_respected(self, runs_dir, monkeypatch):
        """continuation_version passed via context must be used for the run_name suffix."""
        import services.training_autopilot as autopilot

        self._make_improving_run(runs_dir, "run_2024-01-15_1200_base")

        enqueue_mock = MagicMock(return_value={"id": 42, "status": "queued"})
        with patch("jobs.queue_manager.enqueue_train_job", enqueue_mock, create=True):
            monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)
            autopilot.run_autopilot(
                "run_2024-01-15_1200_base",
                context={"continuation_version": 7},
            )

        call_payload = enqueue_mock.call_args[0][0]
        assert call_payload["run_name"] == "base_v7"
        assert call_payload["autopilot_context"]["continuation_version"] == 8


# ---------------------------------------------------------------------------
# _derive_base_run_name (module-level helper)
# ---------------------------------------------------------------------------

class TestDeriveBaseRunName:
    def test_standard_run_id_with_name(self):
        import services.training_autopilot as autopilot
        assert autopilot._derive_base_run_name("run_2024-01-15_1200_myexperiment") == "myexperiment"

    def test_run_id_without_name_returns_empty(self):
        import services.training_autopilot as autopilot
        assert autopilot._derive_base_run_name("run_2024-01-15_1200") == ""

    def test_strips_trailing_v_suffix(self):
        import services.training_autopilot as autopilot
        assert autopilot._derive_base_run_name("run_2024-01-15_1200_myexperiment_v2") == "myexperiment"
        assert autopilot._derive_base_run_name("run_2024-01-15_1200_myexperiment_v10") == "myexperiment"

    def test_non_matching_run_id_returns_empty(self):
        import services.training_autopilot as autopilot
        assert autopilot._derive_base_run_name("arbitrary_string") == ""
        assert autopilot._derive_base_run_name("") == ""

    def test_underscore_in_name_preserved(self):
        import services.training_autopilot as autopilot
        result = autopilot._derive_base_run_name("run_2024-01-15_1200_my_long_name")
        assert result == "my_long_name"


# ---------------------------------------------------------------------------
# Regression branch (stop_regression)
# ---------------------------------------------------------------------------

class TestRegressionBranch:
    def test_reward_drops_more_than_threshold_gives_stop_regression(self, runs_dir):
        """Reward dropping >10% over window must produce action=stop_regression."""
        import services.training_autopilot as autopilot

        # improvement = (-7000 - -5000) / 5000 = -40%  (well below -10% threshold)
        _make_run(
            runs_dir, "run_reg",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5500.0, -6000.0, -7000.0],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_reg")
        assert decision["action"] == "stop_regression"

    def test_regression_next_payload_is_none(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_reg",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5500.0, -6000.0, -7000.0],
        )

        decision = autopilot.analyze_run("run_reg")
        assert decision["next_payload"] is None

    def test_regression_reason_mentions_regress(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_reg",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5500.0, -6000.0, -7000.0],
        )

        decision = autopilot.analyze_run("run_reg")
        assert "regress" in decision["reason"].lower()

    def test_regression_ai_advisor_not_called(self, runs_dir, monkeypatch):
        """stop_regression must never trigger the AI advisor."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_reg",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5500.0, -6000.0, -7000.0],
        )

        ai_mock = MagicMock(return_value={"action": "fine_tune", "reason": "mock", "next_payload": {}, "advisor": "ai"})
        monkeypatch.setattr(autopilot, "_call_ai_advisor", ai_mock)
        monkeypatch.setattr(autopilot, "is_stop_requested", lambda: False)

        autopilot.run_autopilot("run_reg", dry_run=False)
        ai_mock.assert_not_called()

    def test_small_drop_below_threshold_is_not_regression(self, runs_dir):
        """A drop of exactly 5% (< REGRESSION_THRESHOLD=10%) must NOT be stop_regression."""
        import services.training_autopilot as autopilot

        # improvement = (-5250 - -5000) / 5000 = -5%
        _make_run(
            runs_dir, "run_noreg",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5050.0, -5150.0, -5250.0],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_noreg")
        assert decision["action"] != "stop_regression"


# ---------------------------------------------------------------------------
# Bankruptcy improving override (flat reward → continue instead of stop)
# ---------------------------------------------------------------------------

class TestBankruptcyImprovingOverride:
    def test_flat_reward_with_improving_bankruptcy_continues(self, runs_dir):
        """Flat reward + bankruptcy falling ≥5pp must override plateau stop → continue."""
        import services.training_autopilot as autopilot

        # Flat rewards (improvement ~0%), bankruptcy drops from 60% to 30% over window
        _make_run(
            runs_dir, "run_br_improve",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            bankruptcy_rates=[0.60, 0.50, 0.40, 0.30],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_br_improve")
        assert decision["action"] == "continue"

    def test_continuing_on_bankruptcy_improvement_has_next_payload(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_br_improve",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            bankruptcy_rates=[0.60, 0.50, 0.40, 0.30],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_br_improve")
        assert decision["next_payload"] is not None

    def test_reason_mentions_bankruptcy_improvement(self, runs_dir):
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_br_improve",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            bankruptcy_rates=[0.60, 0.50, 0.40, 0.30],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_br_improve")
        assert "bankruptcy" in decision["reason"].lower()

    def test_small_bankruptcy_improvement_below_threshold_still_stops(self, runs_dir):
        """A 3pp drop (< 5pp threshold) must NOT override the plateau stop."""
        import services.training_autopilot as autopilot

        # Bankruptcy drops from 33% to 30% — only 3pp
        _make_run(
            runs_dir, "run_br_small",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            bankruptcy_rates=[0.33, 0.32, 0.31, 0.30],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_br_small")
        assert decision["action"] == "stop"

    def test_bankruptcy_worsening_does_not_override(self, runs_dir):
        """Bankruptcy going up must not trigger the improving override."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_br_worse",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            bankruptcy_rates=[0.30, 0.40, 0.50, 0.60],
            invalid_action_rate=0.03,
        )

        decision = autopilot.analyze_run("run_br_worse")
        assert decision["action"] == "stop"

    def test_metrics_include_bankruptcy_trend_fields(self, runs_dir):
        """analyze_run must always include bankruptcy trend metrics."""
        import services.training_autopilot as autopilot

        _make_run(
            runs_dir, "run_metrics",
            final_epsilon=0.10,
            eval_rewards=[-5000.0, -5010.0, -4995.0, -5002.0],
            bankruptcy_rates=[0.60, 0.50, 0.40, 0.30],
        )

        decision = autopilot.analyze_run("run_metrics")
        assert "bankruptcy_rate_first" in decision["metrics"]
        assert "bankruptcy_rate_last" in decision["metrics"]
        assert "bankruptcy_improving" in decision["metrics"]
        assert decision["metrics"]["bankruptcy_improving"] is True
