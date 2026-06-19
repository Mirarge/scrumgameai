from __future__ import annotations

import json

from .constants import (
    CONTINUE_EPISODES,
    EPSILON_DECAY_MAX,
    EPSILON_DECAY_MIN,
    EPISODES_MAX,
    EPISODES_MIN,
    LR_MAX,
    LR_MIN,
    MAX_AI_INTERVENTIONS,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
)


def call_ai_advisor(
    metrics: dict,
    current_config: dict,
    stop_reason: str,
    intervention_number: int,
    best_checkpoint_path: str | None,
) -> dict:
    """
    Ask the NVIDIA-hosted LLM to suggest one bounded fine-tune adjustment.

    Returns a dict with keys:
      action          "fine_tune" or "stop"
      reason          str
      next_payload    dict | None  (training job payload if action == "fine_tune")
      advisor         "ai"
    """
    try:
        from openai import OpenAI  # optional dependency; fail gracefully
    except ImportError:
        return {
            "action": "stop",
            "reason": "AI advisor unavailable (openai package not installed). Stopping.",
            "next_payload": None,
            "advisor": "ai",
        }

    bankruptcy_trend_str = ""
    if metrics.get("bankruptcy_rate_first") is not None and metrics.get("bankruptcy_rate_last") is not None:
        br_first = metrics["bankruptcy_rate_first"]
        br_last = metrics["bankruptcy_rate_last"]
        direction = "improving" if br_first > br_last else "worsening"
        bankruptcy_trend_str = f"Bankruptcy rate trend       : {br_first:.1%} → {br_last:.1%} ({direction})"
    else:
        bankruptcy_trend_str = "Bankruptcy rate trend       : insufficient data"

    prompt = f"""You are a hyperparameter tuning advisor for a Double DQN reinforcement learning agent
that learns to play a Scrum board game called "Scrum Game".

=== Game context ===
- Each episode is a single game consisting of at most 6 sprints (turns).
- The agent starts with €25,000 and a mandatory loan of €50,000 (€75,000 total).
- Per sprint the agent chooses a product to work on and rolls dice to determine sprint outcome.
- A failed sprint (dice roll too high) costs money proportional to the deviation; a successful sprint earns money.
- Bankruptcy (running out of money mid-game) ends the episode early with a large penalty.
- A "good" trained agent achieves: bankruptcy rate <20%, average ending money >€50,000,
  and average reward >0 (net positive after penalties and loan repayment).

=== Situation ===
The deterministic autopilot has decided to STOP training for the following reason:
"{stop_reason}"

This is AI intervention {intervention_number} of {MAX_AI_INTERVENTIONS}.
After {MAX_AI_INTERVENTIONS} interventions the run must stop regardless of your suggestion.

=== Recent evaluation metrics ===
Evaluation windows analysed : {metrics.get("eval_windows_analyzed")}
Latest average reward        : {metrics.get("latest_reward")} (€; positive = profitable)
Average ending money         : {metrics.get("average_ending_money")} (€; None if unavailable)
Reward improvement ratio     : {metrics.get("reward_improvement_ratio")} (fraction; >0.02 = improving)
Reward coefficient of var.   : {metrics.get("reward_cv")} (>0.20 = high variance)
Bankruptcy rate              : {metrics.get("bankruptcy_rate")} (fraction; target <0.20)
{bankruptcy_trend_str}
Invalid action rate          : {metrics.get("invalid_action_rate")} (fraction; >0.10 = under-explored)

=== Current hyperparameters ===
learning_rate          : {current_config.get("learning_rate")} (safe range: {LR_MIN}–{LR_MAX})
epsilon_decay_end : {current_config.get("epsilon_decay_end")} (safe range: {EPSILON_DECAY_MIN}–{EPSILON_DECAY_MAX})

=== Your task ===
Decide whether ONE specific hyperparameter change is likely to help the agent escape
the current plateau, or whether the run should truly stop.

Guidance:
- If bankruptcy rate is still high (>40%) and reward is flat, the agent may benefit from
  more exploration (increase epsilon_decay_end) or a lower, more stable learning rate.
- If bankruptcy rate is already low (<20%) but reward is flat, the agent may have converged —
  consider stopping unless variance is high (then lower learning_rate first).
- If average_ending_money is positive but reward is flat, the agent may be close to optimal
  and further tuning is unlikely to help much.

Respond with a JSON object and nothing else:
{{
  "action": "fine_tune" or "stop",
  "reason": "<one sentence>",
  "learning_rate": <float between {LR_MIN} and {LR_MAX}, optional>,
  "epsilon_decay_end": <int between {EPSILON_DECAY_MIN} and {EPSILON_DECAY_MAX}, optional>,
  "episodes": <int between {EPISODES_MIN} and {EPISODES_MAX}, default {CONTINUE_EPISODES}>
}}

Only include learning_rate or epsilon_decay_end if you are changing them.
Do not suggest rule changes or anything outside these two hyperparameters."""

    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            top_p=1,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return {
            "action": "stop",
            "reason": f"AI advisor call failed ({exc}). Stopping.",
            "next_payload": None,
            "advisor": "ai",
        }

    # Extract the JSON block from the response
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        suggestion = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "action": "stop",
            "reason": f"AI advisor returned unparseable response. Stopping. Raw: {raw[:200]}",
            "next_payload": None,
            "advisor": "ai",
        }

    ai_action = suggestion.get("action", "stop")
    ai_reason = suggestion.get("reason", "")

    if ai_action != "fine_tune":
        return {
            "action": "stop",
            "reason": f"AI advisor confirmed stop. {ai_reason}",
            "next_payload": None,
            "advisor": "ai",
        }

    # Clamp values to safe bounds
    new_lr = current_config["learning_rate"]
    if "learning_rate" in suggestion:
        new_lr = max(LR_MIN, min(LR_MAX, float(suggestion["learning_rate"])))

    new_epsilon_decay = current_config["epsilon_decay_end"]
    if "epsilon_decay_end" in suggestion:
        new_epsilon_decay = max(
            EPSILON_DECAY_MIN,
            min(EPSILON_DECAY_MAX, float(suggestion["epsilon_decay_end"])),
        )

    episodes = max(EPISODES_MIN, min(EPISODES_MAX, int(suggestion.get("episodes", CONTINUE_EPISODES))))

    return {
        "action": "fine_tune",
        "reason": f"AI advisor (intervention {intervention_number}/{MAX_AI_INTERVENTIONS}): {ai_reason}",
        "next_payload": {
            "episodes": episodes,
            "learning_rate": new_lr,
            "epsilon_decay_end": new_epsilon_decay,
            "resume_from": best_checkpoint_path,
            "resume_mode": "fine-tune",
            "resume_episodes_mode": "incremental",
        },
        "advisor": "ai",
    }


def probe_ai_advisor(metrics: dict | None = None, current_config: dict | None = None) -> dict:
    """
    Call the AI advisor with supplied (or default test) metrics and return its response.
    Use this to verify the NVIDIA API connection and model are working.
    """
    test_metrics = metrics or {
        "eval_windows_analyzed": 4,
        "latest_reward": -12.5,
        "reward_improvement_ratio": 0.005,
        "reward_cv": 0.08,
        "bankruptcy_rate": 0.35,
        "invalid_action_rate": 0.04,
    }
    test_config = current_config or {
        "learning_rate": 0.0005,
        "epsilon_decay_end": 0.9,
    }
    stop_reason = (
        "Reward plateaued (0.5% over 4 windows). Bankruptcy rate: 35.0%."
    )
    result = call_ai_advisor(
        metrics=test_metrics,
        current_config=test_config,
        stop_reason=stop_reason,
        intervention_number=1,
        best_checkpoint_path=None,
    )
    return {"probe": True, "test_metrics": test_metrics, "test_config": test_config, "result": result}
