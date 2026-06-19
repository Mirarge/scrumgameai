from __future__ import annotations

import math
from datetime import datetime, timezone

from services.app_paths import RUNS_DIR
from services.io_utils import read_csv_rows, read_json_or_empty, safe_float

from .constants import (
    BANKRUPTCY_IMPROVEMENT_THRESHOLD,
    CONTINUE_EPISODES,
    EPSILON_EXPLORE_THRESHOLD,
    EPSILON_EXTENSION_FACTOR,
    IMPROVEMENT_MIN_RATIO,
    INVALID_ACTION_HIGH,
    LR_MIN,
    LR_REDUCTION_FACTOR,
    MAX_LR_REDUCTIONS,
    PLATEAU_WINDOW,
    REGRESSION_THRESHOLD,
    VARIANCE_THRESHOLD,
)


def analyze_run(run_id: str, context: dict | None = None) -> dict:
    """
    Analyze a completed run and return an autopilot decision dict.
    Pure analysis — does not enqueue any job or write to disk.

    Decision logic:
      - continue:              reward still improving (>2% over window)
      - lower_lr:             reward improving but variance is high
                              (capped at MAX_LR_REDUCTIONS consecutive reductions)
      - extend_epsilon_decay: reward flat + high invalid action rate
      - stop:                 plateau detected — AI advisor may override this
    """
    context = context or {}
    lr_reduction_count = int(context.get("lr_reduction_count", 0))

    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise ValueError(f"Run `{run_id}` not found.")

    eval_rows = read_csv_rows(run_dir / "reports" / "evaluation_history.csv")
    training_config = read_json_or_empty(run_dir / "training_config.json")

    # Prefer latest_scrum_model.pth (final episode of this run) for continuation so
    # each cycle advances the episode counter rather than looping from the best-reward
    # episode (which may be earlier than where training actually reached).
    latest_checkpoint = run_dir / "checkpoints" / "latest_scrum_model.pth"
    best_checkpoint = run_dir / "checkpoints" / "best_scrum_model.pth"
    if latest_checkpoint.exists():
        best_checkpoint_path = str(latest_checkpoint)
    elif best_checkpoint.exists():
        best_checkpoint_path = str(best_checkpoint)
    else:
        best_checkpoint_path = None

    rewards = [safe_float(r.get("average_reward")) for r in eval_rows]
    rewards = [r for r in rewards if r is not None]

    last_row = eval_rows[-1] if eval_rows else {}
    invalid_action_rate = safe_float(last_row.get("invalid_action_rate")) or 0.0
    bankruptcy_rate = safe_float(last_row.get("bankruptcy_rate")) or 0.0
    average_ending_money = safe_float(last_row.get("average_ending_money"))

    # Bankruptcy rate trend over the plateau window (negative = improving)
    br_window = [safe_float(r.get("bankruptcy_rate")) for r in eval_rows[-PLATEAU_WINDOW:]]
    br_window = [r for r in br_window if r is not None]
    bankruptcy_rate_first = br_window[0] if len(br_window) >= 2 else None
    bankruptcy_rate_last = br_window[-1] if len(br_window) >= 2 else None
    bankruptcy_improving = (
        bankruptcy_rate_first is not None
        and bankruptcy_rate_last is not None
        and (bankruptcy_rate_first - bankruptcy_rate_last) >= BANKRUPTCY_IMPROVEMENT_THRESHOLD
    )

    current_lr = float(training_config.get("learning_rate", 0.0005))
    current_epsilon_decay = float(training_config.get("epsilon_decay_end", 0.9))
    auto_continue_enabled = bool(training_config.get("auto_continue_enabled", False))
    auto_continue_cycles = int(training_config.get("auto_continue_cycles", 0) or 0)

    # Read the latest epsilon from the training log.
    log_rows = read_csv_rows(run_dir / "reports" / "logs.csv")
    latest_log = log_rows[-1] if log_rows else {}
    latest_epsilon = safe_float(latest_log.get("epsilon"))
    if latest_epsilon is None:
        latest_epsilon = 1.0  # assume unexplored if no log yet

    improvement = None
    cv = None

    if latest_epsilon > EPSILON_EXPLORE_THRESHOLD:
        # Agent is still heavily exploring — plateau detection is unreliable.
        action = "continue"
        reason = (
            f"Still in exploration phase (epsilon={latest_epsilon:.3f} > {EPSILON_EXPLORE_THRESHOLD}). "
            f"Continuing without plateau check."
        )
        improvement = None
        cv = None
    elif len(rewards) < PLATEAU_WINDOW:
        action = "continue"
        reason = (
            f"Only {len(rewards)} evaluation windows recorded; "
            f"need {PLATEAU_WINDOW} to assess plateau. Continuing."
        )
        improvement = None
        cv = None
    else:
        window = rewards[-PLATEAU_WINDOW:]
        first, last = window[0], window[-1]
        improvement = (last - first) / abs(first) if first != 0 else 0.0

        mean = sum(window) / len(window)
        std = math.sqrt(sum((r - mean) ** 2 for r in window) / len(window))
        cv = std / abs(mean) if mean != 0 else 0.0

        if improvement < -REGRESSION_THRESHOLD:
            # Active regression: reward dropped significantly — skip AI, stop immediately.
            action = "stop_regression"
            reason = (
                f"Reward actively regressed ({improvement:.1%} over {PLATEAU_WINDOW} windows). "
                f"Bankruptcy rate: {bankruptcy_rate:.1%}. "
                f"Stopping immediately — AI advisor will not be consulted."
            )
        elif improvement > IMPROVEMENT_MIN_RATIO:
            if cv > VARIANCE_THRESHOLD and lr_reduction_count < MAX_LR_REDUCTIONS:
                action = "lower_lr"
                reason = (
                    f"Reward improving ({improvement:.1%} over {PLATEAU_WINDOW} windows) "
                    f"but results are noisy (CV={cv:.2f}). Reducing learning rate "
                    f"(reduction {lr_reduction_count + 1}/{MAX_LR_REDUCTIONS})."
                )
            elif cv > VARIANCE_THRESHOLD:
                # LR reduction cap reached — treat high variance as a plateau signal.
                action = "stop"
                reason = (
                    f"Reward improving ({improvement:.1%}) but variance remains high (CV={cv:.2f}) "
                    f"after {lr_reduction_count} LR reductions. Stopping."
                )
            else:
                action = "continue"
                reason = (
                    f"Reward improving steadily ({improvement:.1%} over {PLATEAU_WINDOW} windows, "
                    f"CV={cv:.2f}). Continuing unchanged."
                )
        else:
            if bankruptcy_improving:
                # Reward flat but bankruptcy rate is falling meaningfully — real progress.
                action = "continue"
                reason = (
                    f"Reward plateaued ({improvement:.1%} over {PLATEAU_WINDOW} windows) "
                    f"but bankruptcy rate improving "
                    f"({bankruptcy_rate_first:.1%} → {bankruptcy_rate_last:.1%}). Continuing."
                )
            elif invalid_action_rate > INVALID_ACTION_HIGH:
                action = "extend_epsilon_decay"
                reason = (
                    f"Reward plateaued ({improvement:.1%} over {PLATEAU_WINDOW} windows) "
                    f"with high invalid action rate ({invalid_action_rate:.1%}). Extending exploration."
                )
            else:
                action = "stop"
                reason = (
                    f"Reward plateaued ({improvement:.1%} over {PLATEAU_WINDOW} windows). "
                    f"Bankruptcy rate: {bankruptcy_rate:.1%}."
                )

    next_payload = None
    if action not in {"stop", "stop_regression"}:
        # Apply LR floor so repeated reductions can't go below the safe minimum.
        new_lr = max(LR_MIN, current_lr * LR_REDUCTION_FACTOR) if action == "lower_lr" else current_lr
        # Epsilon decay: extend by adding episodes on top of the current decay period
        # rather than scaling the absolute value, so the extension is meaningful
        # regardless of where in training the continuation starts.
        new_epsilon_decay = (
            current_epsilon_decay + int(current_epsilon_decay * (EPSILON_EXTENSION_FACTOR - 1.0))
            if action == "extend_epsilon_decay"
            else current_epsilon_decay
        )
        resume_mode = "strict" if action == "continue" else "fine-tune"
        next_payload = {
            "episodes": CONTINUE_EPISODES,
            "learning_rate": new_lr,
            "epsilon_decay_end": new_epsilon_decay,
            "resume_from": best_checkpoint_path,
            "resume_mode": resume_mode,
            "resume_episodes_mode": "incremental",
            "rule_randomization_enabled": bool(training_config.get("rule_randomization_enabled", False)),
            "rule_randomization_frequency": int(training_config.get("rule_randomization_frequency", 1) or 1),
            "rule_randomization_eval_configs": int(training_config.get("rule_randomization_eval_configs", 12) or 12),
            "rule_randomization_bounds": dict(training_config.get("rule_randomization_bounds", {})),
            "auto_continue_enabled": auto_continue_enabled,
            "auto_continue_cycles": auto_continue_cycles,
        }

    return {
        "run_id": run_id,
        "action": action,
        "reason": reason,
        "advisor": "logic",
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "eval_windows_analyzed": len(rewards),
            "latest_reward": rewards[-1] if rewards else None,
            "latest_epsilon": latest_epsilon,
            "reward_improvement_ratio": improvement,
            "reward_cv": cv,
            "bankruptcy_rate": bankruptcy_rate,
            "bankruptcy_rate_first": bankruptcy_rate_first,
            "bankruptcy_rate_last": bankruptcy_rate_last,
            "bankruptcy_improving": bankruptcy_improving,
            "average_ending_money": average_ending_money,
            "invalid_action_rate": invalid_action_rate,
        },
        "current_config": {
            "learning_rate": current_lr,
            "epsilon_decay_end": current_epsilon_decay,
            "rule_randomization_enabled": bool(training_config.get("rule_randomization_enabled", False)),
            "rule_randomization_frequency": int(training_config.get("rule_randomization_frequency", 1) or 1),
            "rule_randomization_eval_configs": int(training_config.get("rule_randomization_eval_configs", 12) or 12),
            "rule_randomization_bounds": dict(training_config.get("rule_randomization_bounds", {})),
            "auto_continue_enabled": auto_continue_enabled,
            "auto_continue_cycles": auto_continue_cycles,
        },
        "context": {
            "lr_reduction_count": lr_reduction_count,
        },
        "best_checkpoint_path": best_checkpoint_path,
        "next_payload": next_payload,
    }
