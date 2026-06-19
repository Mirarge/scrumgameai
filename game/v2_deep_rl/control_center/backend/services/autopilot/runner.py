from __future__ import annotations

from services.app_paths import RUNS_DIR

from .advisor import call_ai_advisor
from .analysis import analyze_run
from .constants import CONTINUE_EPISODES, MAX_AI_INTERVENTIONS
from .history import write_decision_record
from .naming import derive_base_run_name
from .state import clear_stop_request, get_settings, is_stop_requested


def run_autopilot(run_id: str, dry_run: bool = False, context: dict | None = None) -> dict:
    """
    Analyze run, record the decision to disk, and enqueue the next job.

    context is passed forward through job payloads to track state across runs:
      ai_intervention_count  int  how many AI-advised fine-tunes have run so far

    Flow:
      1. Logic classifies the run.
      2. If logic says stop AND intervention count < MAX_AI_INTERVENTIONS:
             AI advisor suggests a fine-tune OR confirms stop.
      3. If user requested stop-after-cycle: override to stop.
      4. Write decision record.
      5. Enqueue next job (if not stopping), carrying context + autopilot_after_completion.
    """
    from jobs.queue_manager import enqueue_train_job  # avoid circular at module load

    context = context or {}
    ai_intervention_count = int(context.get("ai_intervention_count", 0))
    lr_reduction_count = int(context.get("lr_reduction_count", 0))
    # base_run_name is carried forward so the whole chain shares the original name.
    # continuation_version tracks what vN suffix the *next* run should get (starts at 2).
    base_run_name = context.get("base_run_name") or derive_base_run_name(run_id)
    continuation_version = int(context.get("continuation_version", 2))
    auto_continue_cycle_count = int(context.get("auto_continue_cycle_count", 0))

    decision = analyze_run(run_id, context=context)
    current_config = decision.get("current_config", {})
    auto_continue_enabled = bool(current_config.get("auto_continue_enabled", False))
    auto_continue_cycles = int(current_config.get("auto_continue_cycles", 0) or 0)
    auto_continue_remaining = auto_continue_enabled and (
        auto_continue_cycles == 0 or auto_continue_cycle_count < auto_continue_cycles
    )

    if decision["action"] in {"stop", "stop_regression"} and auto_continue_remaining:
        decision["action"] = "continue"
        decision["reason"] = (
            f"Auto-continue cycle {auto_continue_cycle_count + 1}"
            + (f"/{auto_continue_cycles}" if auto_continue_cycles else "")
            + f" requested by training config. Previous reason: {decision['reason']}"
        )
        decision["advisor"] = "auto_continue"
        decision["next_payload"] = {
            "episodes": CONTINUE_EPISODES,
            "learning_rate": current_config.get("learning_rate", 0.0005),
            "epsilon_decay_end": current_config.get("epsilon_decay_end", 0.9),
            "resume_from": decision["best_checkpoint_path"],
            "resume_mode": "strict",
            "resume_episodes_mode": "incremental",
            "rule_randomization_enabled": bool(current_config.get("rule_randomization_enabled", False)),
            "rule_randomization_frequency": int(current_config.get("rule_randomization_frequency", 1) or 1),
            "rule_randomization_eval_configs": int(current_config.get("rule_randomization_eval_configs", 12) or 12),
            "rule_randomization_bounds": dict(current_config.get("rule_randomization_bounds", {})),
            "auto_continue_enabled": auto_continue_enabled,
            "auto_continue_cycles": auto_continue_cycles,
        }

    # --- AI advisor: only when logic says stop (not regression) and budget remains ---
    settings = get_settings()
    ai_enabled = settings.get("ai_enabled", True)
    if not dry_run and decision["action"] == "stop" and ai_enabled and ai_intervention_count < MAX_AI_INTERVENTIONS:
        ai_result = call_ai_advisor(
            metrics=decision["metrics"],
            current_config=decision["current_config"],
            stop_reason=decision["reason"],
            intervention_number=ai_intervention_count + 1,
            best_checkpoint_path=decision["best_checkpoint_path"],
        )
        decision["ai_advisor"] = ai_result
        if ai_result["action"] == "fine_tune":
            decision["action"] = "fine_tune"
            decision["reason"] = ai_result["reason"]
            decision["advisor"] = "ai"
            decision["next_payload"] = ai_result["next_payload"]

    # --- User-requested stop overrides everything except dry_run ---
    if not dry_run and decision["action"] not in {"stop", "stop_regression"} and is_stop_requested():
        decision["action"] = "stop"
        decision["reason"] = "Stop requested by user via stop-after-cycle flag. " + decision["reason"]
        decision["next_payload"] = None
        clear_stop_request()

    if decision.get("next_payload"):
        decision["next_payload"].setdefault(
            "rule_randomization_enabled",
            bool(current_config.get("rule_randomization_enabled", False)),
        )
        decision["next_payload"].setdefault(
            "rule_randomization_frequency",
            int(current_config.get("rule_randomization_frequency", 1) or 1),
        )
        decision["next_payload"].setdefault(
            "rule_randomization_eval_configs",
            int(current_config.get("rule_randomization_eval_configs", 12) or 12),
        )
        decision["next_payload"].setdefault(
            "rule_randomization_bounds",
            dict(current_config.get("rule_randomization_bounds", {})),
        )
        decision["next_payload"].setdefault("auto_continue_enabled", auto_continue_enabled)
        decision["next_payload"].setdefault("auto_continue_cycles", auto_continue_cycles)

    run_dir = RUNS_DIR / run_id

    if dry_run or decision["action"] in {"stop", "stop_regression"} or not decision["next_payload"]:
        decision["job_enqueued"] = False
        # Autopilot truly stopped — auto-queue a final robustness evaluation so the
        # user gets a fresh rating without having to trigger it manually.
        if not dry_run and decision["action"] in {"stop", "stop_regression"}:
            best_pth = run_dir / "checkpoints" / "best_scrum_model.pth"
            if best_pth.exists():
                try:
                    from jobs.queue_manager import enqueue_evaluation_job
                    eval_job = enqueue_evaluation_job({
                        "job_type": "robustness",
                        "run_dir": str(run_dir.resolve()),
                    })
                    decision["auto_eval_job_id"] = eval_job["id"]
                except Exception:
                    pass
        write_decision_record(run_dir, decision)
        if not dry_run and decision["action"] in {"stop", "stop_regression"}:
            try:
                from services.campaign_service import get_campaign_for_run, on_run_stopped

                if get_campaign_for_run(run_id):
                    on_run_stopped(run_id, decision.get("metrics", {}))
            except Exception:
                pass
        return decision

    # Build versioned run name: keep original name + v2, v3, …
    versioned_run_name = f"{base_run_name}_v{continuation_version}" if base_run_name else f"v{continuation_version}"

    # Build next context counters
    next_ai_count = ai_intervention_count + 1 if decision["advisor"] == "ai" else ai_intervention_count
    next_lr_reduction_count = lr_reduction_count + 1 if decision["action"] == "lower_lr" else lr_reduction_count
    next_auto_continue_cycle_count = (
        auto_continue_cycle_count + 1 if auto_continue_enabled else auto_continue_cycle_count
    )

    payload = {
        **decision["next_payload"],
        "run_name": versioned_run_name,
        "autopilot_after_completion": True,
        "autopilot_context": {
            "ai_intervention_count": next_ai_count,
            "lr_reduction_count": next_lr_reduction_count,
            "base_run_name": base_run_name,
            "continuation_version": continuation_version + 1,
            "auto_continue_cycle_count": next_auto_continue_cycle_count,
        },
    }
    job = enqueue_train_job(payload)
    decision["job_enqueued"] = True
    decision["job_id"] = job["id"]
    decision["job_status"] = job["status"]
    write_decision_record(run_dir, decision)
    return decision
