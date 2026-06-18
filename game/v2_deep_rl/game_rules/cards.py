from __future__ import annotations

from dataclasses import dataclass
import random

from config.config_manager import GameConfig, normalize_product_key


@dataclass(frozen=True)
class IncidentCard:
    """One incident card and its environment effect metadata."""

    card_id: int
    name: str
    description: str
    effect_type: str
    target_product_keys: tuple[str, ...] = ()
    delta_money: int = 0
    target_sprint: int | None = None
    set_value_money: int | None = None
    future_only: bool = True
    weight: float = 1.0

    def _scaled_delta(self, env, value: int) -> int:
        return int(round(value * env.incident_severity_multiplier))

    def _scaled_exact_value(self, env, value: int) -> int:
        return int(round(value * env.incident_severity_multiplier))

    def apply_effect(self, env):
        """Apply the incident effect to the environment state."""
        target_product_ids = [
            env.product_lookup[normalize_product_key(product_key)]
            for product_key in self.target_product_keys
            if normalize_product_key(product_key) in env.product_lookup
        ]

        if self.effect_type == "set_future_product_to_zero":
            for product_id in target_product_ids:
                for sprint_id in range(1, env.sprints_per_product + 1):
                    if env.is_sprint_future(product_id, sprint_id, future_only=self.future_only):
                        env.set_incident_value(product_id, sprint_id, 0)
            return

        if self.effect_type == "adjust_future_products":
            scaled_delta = self._scaled_delta(env, self.delta_money)
            for product_id in target_product_ids:
                for sprint_id in range(1, env.sprints_per_product + 1):
                    if env.is_sprint_future(product_id, sprint_id, future_only=self.future_only):
                        env.add_incident_delta(product_id, sprint_id, scaled_delta)
            return

        if self.effect_type == "adjust_specific_sprint_globally":
            scaled_delta = self._scaled_delta(env, self.delta_money)
            for product_id in range(1, env.products_count + 1):
                if env.is_sprint_future(product_id, self.target_sprint, future_only=self.future_only):
                    env.add_incident_delta(product_id, self.target_sprint, scaled_delta)
            return

        if self.effect_type == "set_specific_sprint_exact":
            if self.set_value_money is None:
                raise ValueError("set_specific_sprint_exact requires set_value_money.")
            scaled_value = self._scaled_exact_value(env, self.set_value_money)
            for product_id in target_product_ids:
                if env.is_sprint_future(product_id, self.target_sprint, future_only=self.future_only):
                    env.set_incident_value(product_id, self.target_sprint, scaled_value)
            return

        raise ValueError(f"Unsupported incident effect type: {self.effect_type}")


class IncidentDeck:
    """Deck with draw and discard mechanics for the Scrum Game incidents."""

    def __init__(self, cards):
        self.all_cards = list(cards)
        self.draw_pile = []
        self.discard_pile = []
        self.shuffle()

    def _expanded_card_pool(self):
        expanded = []
        for card in self.all_cards:
            copies = max(1, int(round(card.weight)))
            expanded.extend([card] * copies)
        return expanded

    def shuffle(self):
        """Reset the draw pile from the full card list."""
        self.draw_pile = self._expanded_card_pool()
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def reshuffle_discard_pile(self):
        """Move discard pile back into the draw pile when needed."""
        if not self.discard_pile:
            return
        self.draw_pile = list(self.discard_pile)
        random.shuffle(self.draw_pile)
        self.discard_pile = []

    def draw(self):
        """Draw the next incident card, reshuffling the discard pile if needed."""
        if not self.draw_pile:
            self.reshuffle_discard_pile()
        if not self.draw_pile:
            raise RuntimeError("Incident deck is empty and cannot be reshuffled.")
        card = self.draw_pile.pop()
        self.discard_pile.append(card)
        return card


def build_incident_cards(game_config: GameConfig) -> list[IncidentCard]:
    """Build the configured incident deck from GameConfig."""
    return [
        IncidentCard(
            card_id=card_config.card_id,
            name=card_config.name,
            description=card_config.description,
            effect_type=card_config.effect_type,
            target_product_keys=tuple(card_config.target_products),
            delta_money=card_config.delta_money,
            target_sprint=card_config.target_sprint,
            set_value_money=card_config.set_value_money,
            future_only=card_config.future_only,
            weight=card_config.weight,
        )
        for card_config in game_config.incident.cards
    ]
