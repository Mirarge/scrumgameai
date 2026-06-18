from __future__ import annotations

import random

from config.config_manager import RefinementConfig, normalize_product_key


class ConfiguredRefinementModel:
    """Config-driven refinement logic shared by training and play flows."""

    def __init__(self, refinement_config: RefinementConfig):
        self.refinement_config = refinement_config
        self.rule_lookup = {
            rule.product_key: {
                "increase_rolls": set(rule.increase_rolls),
                "decrease_rolls": set(rule.decrease_rolls),
            }
            for rule in refinement_config.product_rules
        }

    def apply(self, env, product_id):
        """
        Apply one refinement roll to the product played this turn.

        Increase:
        - add one feature to every future sprint of that product

        Decrease:
        - remove one feature from the last future sprint of that product
        """
        product_name = env.product_names[product_id - 1]
        product_key = normalize_product_key(product_name)
        rules = self.rule_lookup.get(
            product_key,
            {"increase_rolls": set(), "decrease_rolls": set()},
        )
        roll = random.randint(1, self.refinement_config.die_sides)

        result = {
            "roll": roll,
            "effect": "none",
            "target_product": product_id,
            "future_sprints_changed": [],
        }

        future_sprints = [
            sprint_id
            for sprint_id in range(1, env.sprints_per_product + 1)
            if env.is_sprint_future(product_id, sprint_id, future_only=True)
        ]

        if not future_sprints:
            return result

        if roll in rules["increase_rolls"]:
            for sprint_id in future_sprints:
                env.add_refinement_delta(product_id, sprint_id, 1)
                result["future_sprints_changed"].append({"sprint_id": sprint_id, "delta": 1})
            result["effect"] = "increase"
            return result

        if roll in rules["decrease_rolls"]:
            last_future_sprint = future_sprints[-1]
            env.add_refinement_delta(product_id, last_future_sprint, -1)
            result["future_sprints_changed"].append({"sprint_id": last_future_sprint, "delta": -1})
            result["effect"] = "decrease"
            return result

        return result
