from __future__ import annotations

import random
from typing import Any

from config.config_manager import GameConfig


DEFAULT_RULE_RANDOMIZATION_BOUNDS: dict[str, Any] = {
    "starting_money": [15000, 45000],
    "max_turns": [4, 10],
    "cost_continue": [0, 2000],
    "cost_switch_mid": [0, 10000],
    "cost_switch_after": [0, 5000],
    "mandatory_loan_amount": [30000, 80000],
    "loan_interest": [1000, 10000],
    "penalty_negative": [500, 2500],
    "penalty_positive": [0, 2500],
    "incident_draw_probability": [0.2, 1.0],
    "incident_severity_multiplier": [0.5, 2.0],
    "dice_sides": [4, 20],
    "dice_count": [1, 4],
}


def _range(bounds: dict[str, Any], key: str) -> tuple[float, float]:
    value = bounds.get(key, DEFAULT_RULE_RANDOMIZATION_BOUNDS[key])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Rule-randomization bound `{key}` must be a two-item range.")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        low, high = high, low
    return low, high


def _rand_int(rng: random.Random, bounds: dict[str, Any], key: str) -> int:
    low, high = _range(bounds, key)
    return int(rng.randint(int(round(low)), int(round(high))))


def _rand_float(rng: random.Random, bounds: dict[str, Any], key: str) -> float:
    low, high = _range(bounds, key)
    return float(rng.uniform(low, high))


def merged_rule_randomization_bounds(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return complete randomization bounds with user overrides applied."""
    result = dict(DEFAULT_RULE_RANDOMIZATION_BOUNDS)
    if overrides:
        result.update(overrides)
    return result


def sample_game_config(
    base_config: GameConfig,
    rng: random.Random,
    bounds: dict[str, Any] | None = None,
    *,
    config_name: str | None = None,
) -> GameConfig:
    """
    Sample a structurally compatible game config for domain-randomized training.

    Product count, sprint count, board shape, and dice-rule count stay fixed so
    the DQN input and output dimensions remain checkpoint-compatible.
    """
    resolved_bounds = merged_rule_randomization_bounds(bounds)
    payload = base_config.to_dict()

    payload["config_name"] = config_name or f"{payload.get('config_name', 'Config')} Randomized"
    payload["starting_money"] = _rand_int(rng, resolved_bounds, "starting_money")
    payload["max_turns"] = _rand_int(rng, resolved_bounds, "max_turns")
    payload["cost_continue"] = _rand_int(rng, resolved_bounds, "cost_continue")
    payload["cost_switch_mid"] = _rand_int(rng, resolved_bounds, "cost_switch_mid")
    payload["cost_switch_after"] = _rand_int(rng, resolved_bounds, "cost_switch_after")
    payload["mandatory_loan_amount"] = _rand_int(rng, resolved_bounds, "mandatory_loan_amount")
    payload["loan_interest"] = _rand_int(rng, resolved_bounds, "loan_interest")
    payload["penalty_negative"] = _rand_int(rng, resolved_bounds, "penalty_negative")
    payload["penalty_positive"] = _rand_int(rng, resolved_bounds, "penalty_positive")

    incident = payload.setdefault("incident", {})
    incident["draw_probability"] = round(_rand_float(rng, resolved_bounds, "incident_draw_probability"), 3)
    incident["severity_multiplier"] = round(_rand_float(rng, resolved_bounds, "incident_severity_multiplier"), 3)

    dice_sides_low, dice_sides_high = _range(resolved_bounds, "dice_sides")
    dice_count_low, dice_count_high = _range(resolved_bounds, "dice_count")
    for rule in payload.get("dice_rules", []):
        rule["dice_sides"] = int(rng.randint(int(round(dice_sides_low)), int(round(dice_sides_high))))
        rule["dice_count"] = int(rng.randint(int(round(dice_count_low)), int(round(dice_count_high))))

    return GameConfig.from_dict(payload)
