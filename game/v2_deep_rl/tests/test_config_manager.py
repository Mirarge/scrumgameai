"""
Tests for config/config_manager.py.

config_manager has no torch dependency so it can be imported directly.
conftest.py adds ENGINE_DIR to sys.path, so `import config.config_manager` works.
"""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

import config.config_manager as cm


# ---------------------------------------------------------------------------
# Minimal valid GameConfig payload (used across multiple tests)
# ---------------------------------------------------------------------------

def _minimal_game_payload(**overrides) -> dict:
    """Return the smallest dict that passes validate_game_config."""
    payload = {
        "schema_version": "1.0",
        "config_name": "Test Config",
        "config_description": "A minimal config for testing",
        "players_count": 2,
        "product_names": ["Alpha", "Beta"],
        "max_turns": 6,
        "starting_money": 25000,
        "ring_value": 5000,
        "cost_continue": 0,
        "cost_switch_mid": 5000,
        "cost_switch_after": 0,
        "mandatory_loan_amount": 50000,
        "loan_interest": 5000,
        "penalty_negative": 1000,
        "penalty_positive": 5000,
        "daily_scrums_per_sprint": 5,
        "daily_scrum_target": 12,
        "board_ring_values": [[10000, 20000, 30000], [15000, 25000, 35000]],
        "board_features": [[1, 2, 3], [2, 3, 1]],
        "dice_rules": [
            {"min_features": 1, "max_features": 1, "dice_count": 1, "dice_sides": 20},
            {"min_features": 2, "max_features": None, "dice_count": 2, "dice_sides": 10},
        ],
        "refinement": {
            "active": True,
            "model_name": "Custom",
            "die_sides": 20,
            "product_rules": [],
        },
        "incident": {
            "active": False,
            "allow_player_specific_incidents": False,
            "draw_probability": 1.0,
            "severity_multiplier": 1.0,
            "cards": [],
        },
        "reserved_fields": {},
    }
    payload.update(overrides)
    return payload


def _make_game_config(**overrides) -> cm.GameConfig:
    return cm.GameConfig.from_dict(_minimal_game_payload(**overrides))


# ---------------------------------------------------------------------------
# DiceRuleConfig
# ---------------------------------------------------------------------------

class TestDiceRuleConfig:
    def test_from_dict_roundtrip(self):
        d = {"min_features": 2, "max_features": 4, "dice_count": 3, "dice_sides": 6}
        rule = cm.DiceRuleConfig.from_dict(d)
        assert rule.to_dict() == d

    def test_max_features_none_roundtrip(self):
        d = {"min_features": 3, "max_features": None, "dice_count": 2, "dice_sides": 10}
        rule = cm.DiceRuleConfig.from_dict(d)
        assert rule.max_features is None
        assert rule.to_dict()["max_features"] is None

    def test_matches_within_range(self):
        rule = cm.DiceRuleConfig(min_features=2, max_features=4, dice_count=2, dice_sides=6)
        assert rule.matches(2)
        assert rule.matches(3)
        assert rule.matches(4)

    def test_matches_unbounded_upper(self):
        rule = cm.DiceRuleConfig(min_features=3, max_features=None, dice_count=2, dice_sides=6)
        assert rule.matches(3)
        assert rule.matches(100)

    def test_does_not_match_below_min(self):
        rule = cm.DiceRuleConfig(min_features=2, max_features=4, dice_count=2, dice_sides=6)
        assert not rule.matches(1)

    def test_does_not_match_above_max(self):
        rule = cm.DiceRuleConfig(min_features=2, max_features=4, dice_count=2, dice_sides=6)
        assert not rule.matches(5)


# ---------------------------------------------------------------------------
# RefinementRuleConfig
# ---------------------------------------------------------------------------

class TestRefinementRuleConfig:
    def test_from_dict_roundtrip(self):
        d = {
            "product_key": "blue",
            "increase_rolls": [1, 2, 3],
            "decrease_rolls": [19, 20],
        }
        rule = cm.RefinementRuleConfig.from_dict(d)
        assert rule.product_key == "blue"
        assert rule.increase_rolls == (1, 2, 3)
        assert rule.decrease_rolls == (19, 20)
        rt = rule.to_dict()
        assert rt["increase_rolls"] == [1, 2, 3]
        assert rt["decrease_rolls"] == [19, 20]

    def test_empty_rolls_default(self):
        d = {"product_key": "red", "increase_rolls": [], "decrease_rolls": []}
        rule = cm.RefinementRuleConfig.from_dict(d)
        assert rule.increase_rolls == ()
        assert rule.decrease_rolls == ()


# ---------------------------------------------------------------------------
# IncidentCardConfig
# ---------------------------------------------------------------------------

class TestIncidentCardConfig:
    def test_from_dict_roundtrip(self):
        d = {
            "card_id": 401,
            "name": "Test Card",
            "description": "A test incident.",
            "effect_type": "adjust_future_products",
            "target_products": ["alpha", "beta"],
            "delta_money": -5000,
            "target_sprint": None,
            "set_value_money": None,
            "future_only": True,
            "weight": 1.5,
        }
        card = cm.IncidentCardConfig.from_dict(d)
        rt = card.to_dict()
        assert rt["card_id"] == 401
        assert rt["target_products"] == ["alpha", "beta"]
        assert rt["weight"] == 1.5
        assert rt["future_only"] is True

    def test_target_sprint_none_roundtrip(self):
        d = {
            "card_id": 402,
            "name": "Card",
            "description": "",
            "effect_type": "set_future_product_to_zero",
            "target_products": [],
        }
        card = cm.IncidentCardConfig.from_dict(d)
        assert card.target_sprint is None
        assert card.to_dict()["target_sprint"] is None


# ---------------------------------------------------------------------------
# GameConfig
# ---------------------------------------------------------------------------

class TestGameConfig:
    def test_from_dict_roundtrip(self):
        payload = _minimal_game_payload()
        gc = cm.GameConfig.from_dict(payload)
        rt = gc.to_dict()

        assert rt["config_name"] == "Test Config"
        assert rt["players_count"] == 2
        assert rt["product_names"] == ["Alpha", "Beta"]
        assert rt["max_turns"] == 6
        assert rt["board_ring_values"] == [[10000, 20000, 30000], [15000, 25000, 35000]]

    def test_to_dict_serializable_to_json(self):
        gc = _make_game_config()
        # Should not raise
        json.dumps(gc.to_dict())

    def test_products_count_property(self):
        gc = _make_game_config()
        assert gc.products_count == 2

    def test_sprints_per_product_property(self):
        gc = _make_game_config()
        assert gc.sprints_per_product == 3

    def test_from_dict_preserves_reserved_fields(self):
        payload = _minimal_game_payload(reserved_fields={"custom_key": "custom_value"})
        gc = cm.GameConfig.from_dict(payload)
        assert gc.reserved_fields["custom_key"] == "custom_value"

    def test_rule_payload_omits_metadata_fields(self):
        gc = _make_game_config(config_name="NameThatShouldBeRemoved")
        rp = gc.rule_payload()
        assert "config_name" not in rp
        assert "config_description" not in rp
        assert "reserved_fields" not in rp

    def test_rule_payload_retains_rule_fields(self):
        gc = _make_game_config()
        rp = gc.rule_payload()
        assert "max_turns" in rp
        assert "board_ring_values" in rp
        assert "dice_rules" in rp

    # --- validate_game_config via from_dict ---

    def test_missing_product_names_raises(self):
        payload = _minimal_game_payload(product_names=[])
        with pytest.raises(ValueError, match="at least one product"):
            cm.GameConfig.from_dict(payload)

    def test_players_count_zero_raises(self):
        payload = _minimal_game_payload(players_count=0)
        with pytest.raises(ValueError, match="players_count"):
            cm.GameConfig.from_dict(payload)

    def test_max_turns_zero_raises(self):
        payload = _minimal_game_payload(max_turns=0)
        with pytest.raises(ValueError, match="max_turns"):
            cm.GameConfig.from_dict(payload)

    def test_mismatched_board_rows_raises(self):
        # Three product names but only two board rows
        payload = _minimal_game_payload(
            product_names=["A", "B", "C"],
            board_ring_values=[[1, 2], [3, 4]],
            board_features=[[1, 2], [3, 4]],
        )
        with pytest.raises(ValueError):
            cm.GameConfig.from_dict(payload)

    def test_missing_dice_rules_raises(self):
        payload = _minimal_game_payload(dice_rules=[])
        with pytest.raises(ValueError, match="dice rule"):
            cm.GameConfig.from_dict(payload)

    def test_schema_version_defaults_to_1_0(self):
        payload = _minimal_game_payload()
        del payload["schema_version"]
        gc = cm.GameConfig.from_dict(payload)
        assert gc.schema_version == "1.0"

    def test_config_name_defaults_when_missing(self):
        payload = _minimal_game_payload()
        del payload["config_name"]
        gc = cm.GameConfig.from_dict(payload)
        assert "Unnamed" in gc.config_name


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_from_dict_roundtrip(self):
        payload = {
            "episodes": 100000,
            "evaluation_episodes": 50,
            "checkpoint_interval": 5000,
            "evaluation_interval": 5000,
            "learning_rate": 0.001,
            "gamma": 0.9,
            "replay_capacity": 50000,
            "batch_size": 64,
            "target_update_frequency": 1000,
            "seed": 7,
            "epsilon_start": 0.8,
            "epsilon_min": 0.02,
            "epsilon_decay_end": 0.9,
            "run_notes": "test run",
            "rule_randomization_enabled": True,
            "rule_randomization_frequency": 2,
            "rule_randomization_eval_configs": 6,
            "rule_randomization_bounds": {"max_turns": [4, 8]},
            "auto_continue_enabled": True,
            "auto_continue_cycles": 3,
        }
        tc = cm.TrainingConfig.from_dict(payload)
        rt = tc.to_dict()
        assert rt == payload

    def test_defaults_when_empty_dict(self):
        tc = cm.TrainingConfig.from_dict({})
        assert tc.episodes == 500000
        assert tc.learning_rate == 0.0005
        assert tc.gamma == 0.85
        assert tc.seed == 42
        assert tc.epsilon_start == 1.0
        assert tc.epsilon_min == 0.05
        assert tc.epsilon_decay_end == 0.9
        assert tc.run_notes == ""
        assert tc.rule_randomization_enabled is False
        assert tc.auto_continue_enabled is False

    def test_to_dict_serializable_to_json(self):
        tc = cm.TrainingConfig.from_dict({})
        json.dumps(tc.to_dict())

    def test_signature_payload_excludes_run_notes(self):
        tc = cm.TrainingConfig.from_dict({"run_notes": "should be excluded"})
        sp = tc.signature_payload()
        assert "run_notes" not in sp

    def test_signature_payload_retains_hyperparams(self):
        tc = cm.TrainingConfig.from_dict({"learning_rate": 0.001})
        sp = tc.signature_payload()
        assert "learning_rate" in sp
        assert sp["learning_rate"] == 0.001

    def test_partial_override_keeps_other_defaults(self):
        tc = cm.TrainingConfig.from_dict({"episodes": 1000})
        assert tc.episodes == 1000
        assert tc.gamma == 0.85  # default unchanged


# ---------------------------------------------------------------------------
# compute_rule_signature
# ---------------------------------------------------------------------------

class TestComputeRuleSignature:
    def test_same_config_same_signature(self):
        gc1 = _make_game_config()
        gc2 = _make_game_config()
        assert cm.compute_rule_signature(gc1) == cm.compute_rule_signature(gc2)

    def test_different_ring_value_different_signature(self):
        gc1 = _make_game_config(ring_value=5000)
        gc2 = _make_game_config(ring_value=10000)
        assert cm.compute_rule_signature(gc1) != cm.compute_rule_signature(gc2)

    def test_different_board_values_different_signature(self):
        gc1 = _make_game_config(board_ring_values=[[1, 2, 3], [4, 5, 6]])
        gc2 = _make_game_config(board_ring_values=[[1, 2, 3], [4, 5, 7]])
        assert cm.compute_rule_signature(gc1) != cm.compute_rule_signature(gc2)

    def test_signature_is_hex_string(self):
        gc = _make_game_config()
        sig = cm.compute_rule_signature(gc)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex
        int(sig, 16)  # must be valid hex

    def test_config_name_change_does_not_change_rule_signature(self):
        """rule_payload omits config_name — so renaming must not affect the signature."""
        gc1 = _make_game_config(config_name="Version A")
        gc2 = _make_game_config(config_name="Version B")
        assert cm.compute_rule_signature(gc1) == cm.compute_rule_signature(gc2)

    def test_different_max_turns_different_signature(self):
        gc1 = _make_game_config(max_turns=6)
        gc2 = _make_game_config(max_turns=8)
        assert cm.compute_rule_signature(gc1) != cm.compute_rule_signature(gc2)


# ---------------------------------------------------------------------------
# compute_training_signature
# ---------------------------------------------------------------------------

class TestComputeTrainingSignature:
    def test_same_config_same_signature(self):
        tc1 = cm.TrainingConfig.from_dict({})
        tc2 = cm.TrainingConfig.from_dict({})
        assert cm.compute_training_signature(tc1) == cm.compute_training_signature(tc2)

    def test_different_lr_different_signature(self):
        tc1 = cm.TrainingConfig.from_dict({"learning_rate": 0.0005})
        tc2 = cm.TrainingConfig.from_dict({"learning_rate": 0.001})
        assert cm.compute_training_signature(tc1) != cm.compute_training_signature(tc2)

    def test_run_notes_change_does_not_change_signature(self):
        """signature_payload omits run_notes — notes must not affect the signature."""
        tc1 = cm.TrainingConfig.from_dict({"run_notes": "run A"})
        tc2 = cm.TrainingConfig.from_dict({"run_notes": "run B"})
        assert cm.compute_training_signature(tc1) == cm.compute_training_signature(tc2)

    def test_signature_is_hex_string(self):
        tc = cm.TrainingConfig.from_dict({})
        sig = cm.compute_training_signature(tc)
        assert isinstance(sig, str)
        assert len(sig) == 64
        int(sig, 16)

    def test_different_gamma_different_signature(self):
        tc1 = cm.TrainingConfig.from_dict({"gamma": 0.85})
        tc2 = cm.TrainingConfig.from_dict({"gamma": 0.99})
        assert cm.compute_training_signature(tc1) != cm.compute_training_signature(tc2)

    def test_different_seed_different_signature(self):
        tc1 = cm.TrainingConfig.from_dict({"seed": 42})
        tc2 = cm.TrainingConfig.from_dict({"seed": 99})
        assert cm.compute_training_signature(tc1) != cm.compute_training_signature(tc2)


# ---------------------------------------------------------------------------
# normalize_product_key
# ---------------------------------------------------------------------------

class TestNormalizeProductKey:
    def test_lowercase_letters_only(self):
        assert cm.normalize_product_key("Blue") == "blue"

    def test_strips_spaces(self):
        assert cm.normalize_product_key("  Red  ") == "red"

    def test_removes_special_chars(self):
        assert cm.normalize_product_key("Hello World!") == "helloworld"

    def test_already_normalized(self):
        assert cm.normalize_product_key("orange") == "orange"

    def test_mixed_case_alphanumeric(self):
        assert cm.normalize_product_key("Product1") == "product1"
