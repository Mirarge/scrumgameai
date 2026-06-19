from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "configs"
DEFAULT_GAME_CONFIG_PATH = CONFIG_DIR / "default_game_config.json"
DEFAULT_TRAINING_CONFIG_PATH = CONFIG_DIR / "default_training_config.json"


def normalize_product_key(value: str) -> str:
    """Convert a display label into a stable lower-case product key."""
    return "".join(character.lower() for character in str(value).strip() if character.isalnum())


@dataclass(frozen=True)
class DiceRuleConfig:
    """One dice regime applied to a range of feature counts."""

    min_features: int
    max_features: int | None
    dice_count: int
    dice_sides: int

    def matches(self, feature_count: int) -> bool:
        upper_ok = self.max_features is None or feature_count <= self.max_features
        return feature_count >= self.min_features and upper_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_features": self.min_features,
            "max_features": self.max_features,
            "dice_count": self.dice_count,
            "dice_sides": self.dice_sides,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiceRuleConfig":
        return cls(
            min_features=int(payload["min_features"]),
            max_features=None if payload.get("max_features") is None else int(payload["max_features"]),
            dice_count=int(payload["dice_count"]),
            dice_sides=int(payload["dice_sides"]),
        )


@dataclass(frozen=True)
class RefinementRuleConfig:
    """Refinement roll behavior for one product key."""

    product_key: str
    increase_rolls: tuple[int, ...]
    decrease_rolls: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_key": self.product_key,
            "increase_rolls": list(self.increase_rolls),
            "decrease_rolls": list(self.decrease_rolls),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefinementRuleConfig":
        return cls(
            product_key=str(payload["product_key"]),
            increase_rolls=tuple(int(value) for value in payload.get("increase_rolls", [])),
            decrease_rolls=tuple(int(value) for value in payload.get("decrease_rolls", [])),
        )


@dataclass(frozen=True)
class RefinementConfig:
    """All refinement behavior shared by the environment."""

    active: bool = True
    model_name: str = "Standard (ID 301)"
    die_sides: int = 20
    product_rules: tuple[RefinementRuleConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "model_name": self.model_name,
            "die_sides": self.die_sides,
            "product_rules": [rule.to_dict() for rule in self.product_rules],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RefinementConfig":
        return cls(
            active=bool(payload.get("active", True)),
            model_name=str(payload.get("model_name", "Standard (ID 301)")),
            die_sides=int(payload.get("die_sides", 20)),
            product_rules=tuple(
                RefinementRuleConfig.from_dict(rule_payload)
                for rule_payload in payload.get("product_rules", [])
            ),
        )


@dataclass(frozen=True)
class IncidentCardConfig:
    """Serialized incident-card definition."""

    card_id: int
    name: str
    description: str
    effect_type: str
    target_products: tuple[str, ...] = ()
    delta_money: int = 0
    target_sprint: int | None = None
    set_value_money: int | None = None
    future_only: bool = True
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "name": self.name,
            "description": self.description,
            "effect_type": self.effect_type,
            "target_products": list(self.target_products),
            "delta_money": self.delta_money,
            "target_sprint": self.target_sprint,
            "set_value_money": self.set_value_money,
            "future_only": self.future_only,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IncidentCardConfig":
        return cls(
            card_id=int(payload["card_id"]),
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            effect_type=str(payload["effect_type"]),
            target_products=tuple(str(value) for value in payload.get("target_products", [])),
            delta_money=int(payload.get("delta_money", 0)),
            target_sprint=None if payload.get("target_sprint") is None else int(payload["target_sprint"]),
            set_value_money=(
                None if payload.get("set_value_money") is None else int(payload["set_value_money"])
            ),
            future_only=bool(payload.get("future_only", True)),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class IncidentConfig:
    """Incident-draw behavior and the configured deck."""

    active: bool = True
    allow_player_specific_incidents: bool = False
    draw_probability: float = 1.0
    severity_multiplier: float = 1.0
    cards: tuple[IncidentCardConfig, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "allow_player_specific_incidents": self.allow_player_specific_incidents,
            "draw_probability": self.draw_probability,
            "severity_multiplier": self.severity_multiplier,
            "cards": [card.to_dict() for card in self.cards],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IncidentConfig":
        return cls(
            active=bool(payload.get("active", True)),
            allow_player_specific_incidents=bool(payload.get("allow_player_specific_incidents", False)),
            draw_probability=float(payload.get("draw_probability", 1.0)),
            severity_multiplier=float(payload.get("severity_multiplier", 1.0)),
            cards=tuple(
                IncidentCardConfig.from_dict(card_payload) for card_payload in payload.get("cards", [])
            ),
        )


@dataclass(frozen=True)
class GameConfig:
    """Canonical rule configuration for one Scrum Game variant."""

    schema_version: str
    config_name: str
    config_description: str
    players_count: int
    product_names: tuple[str, ...]
    max_turns: int
    starting_money: int
    ring_value: int
    cost_continue: int
    cost_switch_mid: int
    cost_switch_after: int
    mandatory_loan_amount: int
    loan_interest: int
    penalty_negative: int
    penalty_positive: int
    daily_scrums_per_sprint: int
    daily_scrum_target: int
    board_ring_values: tuple[tuple[int, ...], ...]
    board_features: tuple[tuple[int, ...], ...]
    dice_rules: tuple[DiceRuleConfig, ...]
    refinement: RefinementConfig
    incident: IncidentConfig
    reserved_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def products_count(self) -> int:
        return len(self.product_names)

    @property
    def sprints_per_product(self) -> int:
        return len(self.board_ring_values[0]) if self.board_ring_values else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_name": self.config_name,
            "config_description": self.config_description,
            "players_count": self.players_count,
            "product_names": list(self.product_names),
            "max_turns": self.max_turns,
            "starting_money": self.starting_money,
            "ring_value": self.ring_value,
            "cost_continue": self.cost_continue,
            "cost_switch_mid": self.cost_switch_mid,
            "cost_switch_after": self.cost_switch_after,
            "mandatory_loan_amount": self.mandatory_loan_amount,
            "loan_interest": self.loan_interest,
            "penalty_negative": self.penalty_negative,
            "penalty_positive": self.penalty_positive,
            "daily_scrums_per_sprint": self.daily_scrums_per_sprint,
            "daily_scrum_target": self.daily_scrum_target,
            "board_ring_values": [list(row) for row in self.board_ring_values],
            "board_features": [list(row) for row in self.board_features],
            "dice_rules": [rule.to_dict() for rule in self.dice_rules],
            "refinement": self.refinement.to_dict(),
            "incident": self.incident.to_dict(),
            "reserved_fields": self.reserved_fields,
        }

    def rule_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("config_name", None)
        payload.pop("config_description", None)
        payload.pop("reserved_fields", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameConfig":
        config = cls(
            schema_version=str(payload.get("schema_version", "1.0")),
            config_name=str(payload.get("config_name", "Unnamed Config")),
            config_description=str(payload.get("config_description", "")),
            players_count=int(payload.get("players_count", 1)),
            product_names=tuple(str(name) for name in payload["product_names"]),
            max_turns=int(payload["max_turns"]),
            starting_money=int(payload["starting_money"]),
            ring_value=int(payload["ring_value"]),
            cost_continue=int(payload.get("cost_continue", 0)),
            cost_switch_mid=int(payload["cost_switch_mid"]),
            cost_switch_after=int(payload["cost_switch_after"]),
            mandatory_loan_amount=int(payload["mandatory_loan_amount"]),
            loan_interest=int(payload["loan_interest"]),
            penalty_negative=int(payload["penalty_negative"]),
            penalty_positive=int(payload["penalty_positive"]),
            daily_scrums_per_sprint=int(payload["daily_scrums_per_sprint"]),
            daily_scrum_target=int(payload["daily_scrum_target"]),
            board_ring_values=tuple(
                tuple(int(value) for value in row) for row in payload["board_ring_values"]
            ),
            board_features=tuple(
                tuple(int(value) for value in row) for row in payload["board_features"]
            ),
            dice_rules=tuple(
                DiceRuleConfig.from_dict(rule_payload)
                for rule_payload in payload.get("dice_rules", [])
            ),
            refinement=RefinementConfig.from_dict(payload.get("refinement", {})),
            incident=IncidentConfig.from_dict(payload.get("incident", {})),
            reserved_fields=dict(payload.get("reserved_fields", {})),
        )
        validate_game_config(config)
        return config


@dataclass(frozen=True)
class TrainingConfig:
    """Canonical training hyperparameters and runtime settings."""

    episodes: int = 500000
    evaluation_episodes: int = 100
    checkpoint_interval: int = 10000
    evaluation_interval: int = 10000
    learning_rate: float = 0.0005
    gamma: float = 0.85
    replay_capacity: int = 100000
    batch_size: int = 128
    target_update_frequency: int = 2000
    seed: int = 42
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_end: float = 0.9
    run_notes: str = ""
    rule_randomization_enabled: bool = False
    rule_randomization_frequency: int = 1
    rule_randomization_eval_configs: int = 12
    rule_randomization_bounds: dict[str, Any] = field(default_factory=dict)
    auto_continue_enabled: bool = False
    auto_continue_cycles: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "evaluation_episodes": self.evaluation_episodes,
            "checkpoint_interval": self.checkpoint_interval,
            "evaluation_interval": self.evaluation_interval,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "target_update_frequency": self.target_update_frequency,
            "seed": self.seed,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay_end": self.epsilon_decay_end,
            "run_notes": self.run_notes,
            "rule_randomization_enabled": self.rule_randomization_enabled,
            "rule_randomization_frequency": self.rule_randomization_frequency,
            "rule_randomization_eval_configs": self.rule_randomization_eval_configs,
            "rule_randomization_bounds": self.rule_randomization_bounds,
            "auto_continue_enabled": self.auto_continue_enabled,
            "auto_continue_cycles": self.auto_continue_cycles,
        }

    def signature_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("run_notes", None)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        return cls(
            episodes=int(payload.get("episodes", 500000)),
            evaluation_episodes=int(payload.get("evaluation_episodes", 100)),
            checkpoint_interval=int(payload.get("checkpoint_interval", 10000)),
            evaluation_interval=int(payload.get("evaluation_interval", 10000)),
            learning_rate=float(payload.get("learning_rate", 0.0005)),
            gamma=float(payload.get("gamma", 0.85)),
            replay_capacity=int(payload.get("replay_capacity", 100000)),
            batch_size=int(payload.get("batch_size", 128)),
            target_update_frequency=int(payload.get("target_update_frequency", 2000)),
            seed=int(payload.get("seed", 42)),
            epsilon_start=float(payload.get("epsilon_start", 1.0)),
            epsilon_min=float(payload.get("epsilon_min", 0.05)),
            epsilon_decay_end=float(payload.get("epsilon_decay_end", 0.9)),
            run_notes=str(payload.get("run_notes", "")),
            rule_randomization_enabled=bool(payload.get("rule_randomization_enabled", False)),
            rule_randomization_frequency=max(1, int(payload.get("rule_randomization_frequency", 1))),
            rule_randomization_eval_configs=max(1, int(payload.get("rule_randomization_eval_configs", 12))),
            rule_randomization_bounds=dict(payload.get("rule_randomization_bounds", {})),
            auto_continue_enabled=bool(payload.get("auto_continue_enabled", False)),
            auto_continue_cycles=max(0, int(payload.get("auto_continue_cycles", 0))),
        )


def _read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _hash_payload(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_rule_signature(game_config: GameConfig) -> str:
    """Return a stable hash for the rule-defining parts of a game config."""
    return _hash_payload(game_config.rule_payload())


def compute_training_signature(training_config: TrainingConfig) -> str:
    """Return a stable hash for the training hyperparameters."""
    return _hash_payload(training_config.signature_payload())


def validate_game_config(game_config: GameConfig) -> None:
    """Raise ValueError when a game config is structurally invalid."""
    if not game_config.product_names:
        raise ValueError("GameConfig requires at least one product.")
    if game_config.players_count < 1:
        raise ValueError("players_count must be at least 1.")
    if game_config.max_turns < 1:
        raise ValueError("max_turns must be at least 1.")
    if not game_config.board_ring_values or not game_config.board_features:
        raise ValueError("Board ring values and board features must be present.")
    if len(game_config.board_ring_values) != len(game_config.product_names):
        raise ValueError("board_ring_values row count must match product_names length.")
    if len(game_config.board_features) != len(game_config.product_names):
        raise ValueError("board_features row count must match product_names length.")
    if game_config.sprints_per_product < 1:
        raise ValueError("Each product must expose at least one sprint.")

    for row in game_config.board_ring_values:
        if len(row) != game_config.sprints_per_product:
            raise ValueError("All board_ring_values rows must have the same sprint count.")
    for row in game_config.board_features:
        if len(row) != game_config.sprints_per_product:
            raise ValueError("All board_features rows must have the same sprint count.")

    if not game_config.dice_rules:
        raise ValueError("At least one dice rule is required.")
    if game_config.mandatory_loan_amount <= 0:
        raise ValueError("mandatory_loan_amount must be positive to avoid infinite loan loops.")
    for rule in game_config.dice_rules:
        if rule.dice_count < 1:
            raise ValueError(f"dice_count must be at least 1 (got {rule.dice_count}).")
        if rule.dice_sides < 2:
            raise ValueError(f"dice_sides must be at least 2 (got {rule.dice_sides}).")

    normalized_products = {normalize_product_key(name) for name in game_config.product_names}
    for rule in game_config.refinement.product_rules:
        if rule.product_key not in normalized_products:
            raise ValueError(f"Unknown refinement product key: {rule.product_key}")
    for card in game_config.incident.cards:
        for product_key in card.target_products:
            if normalize_product_key(product_key) not in normalized_products:
                raise ValueError(f"Unknown incident target product: {product_key}")


def load_game_config(path: str | Path | None = None) -> GameConfig:
    """Load a game config from disk, defaulting to the canonical bundled config."""
    config_path = DEFAULT_GAME_CONFIG_PATH if path is None else Path(path)
    payload = _read_json_file(config_path)
    if "schema_version" in payload and "product_names" in payload:
        return GameConfig.from_dict(payload)
    return map_prototype_to_config(payload)


def save_game_config(game_config: GameConfig, path: str | Path) -> None:
    """Persist a game config as JSON."""
    _write_json_file(Path(path), game_config.to_dict())


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Load a training config from disk, defaulting to the canonical bundled config."""
    config_path = DEFAULT_TRAINING_CONFIG_PATH if path is None else Path(path)
    return TrainingConfig.from_dict(_read_json_file(config_path))


def save_training_config(training_config: TrainingConfig, path: str | Path) -> None:
    """Persist a training config as JSON."""
    _write_json_file(Path(path), training_config.to_dict())


def build_default_refinement_rules(
    product_names: list[str],
    increase_range: tuple[int, int] | None = None,
    decrease_range: tuple[int, int] | None = None,
    model_name: str = "Standard (ID 301)",
) -> tuple[RefinementRuleConfig, ...]:
    """Create refinement rules either from the classical mapping or one shared range."""
    product_keys = [normalize_product_key(name) for name in product_names]
    classical_rules = {
        "yellow": {"increase_rolls": (1, 2), "decrease_rolls": (19, 20)},
        "red": {"increase_rolls": (1, 2), "decrease_rolls": (19, 20)},
        "orange": {"increase_rolls": (1, 2, 3), "decrease_rolls": (19, 20)},
        "green": {"increase_rolls": (1, 2, 3), "decrease_rolls": (19, 20)},
        "purple": {"increase_rolls": (1, 2, 3), "decrease_rolls": (19, 20)},
        "blue": {"increase_rolls": (1, 2, 3, 4), "decrease_rolls": (19, 20)},
        "black": {"increase_rolls": (1,), "decrease_rolls": (20,)},
    }

    if (
        model_name == "Standard (ID 301)"
        and all(product_key in classical_rules for product_key in product_keys)
    ):
        return tuple(
            RefinementRuleConfig(
                product_key=product_key,
                increase_rolls=tuple(classical_rules[product_key]["increase_rolls"]),
                decrease_rolls=tuple(classical_rules[product_key]["decrease_rolls"]),
            )
            for product_key in product_keys
        )

    if increase_range is None:
        increase_range = (1, 2)
    if decrease_range is None:
        decrease_range = (19, 20)

    increase_rolls = tuple(range(increase_range[0], increase_range[1] + 1))
    decrease_rolls = tuple(range(decrease_range[0], decrease_range[1] + 1))
    return tuple(
        RefinementRuleConfig(
            product_key=product_key,
            increase_rolls=increase_rolls,
            decrease_rolls=decrease_rolls,
        )
        for product_key in product_keys
    )


def build_default_incident_cards(product_names: list[str]) -> tuple[IncidentCardConfig, ...]:
    """Return the documented classical incident cards using the current product names."""
    normalized_names = [normalize_product_key(name) for name in product_names]
    classical_keys = {"yellow", "blue", "red", "orange", "green", "purple", "black"}

    if classical_keys.issubset(set(normalized_names)):
        return (
            IncidentCardConfig(
                card_id=401,
                name="Demand Collapse Red",
                description="All future red sprints are worth zero.",
                effect_type="set_future_product_to_zero",
                target_products=("red",),
            ),
            IncidentCardConfig(
                card_id=402,
                name="New Competitors",
                description="Orange and blue products lose value due to new competitors.",
                effect_type="adjust_future_products",
                target_products=("orange", "blue"),
                delta_money=-5000,
            ),
            IncidentCardConfig(
                card_id=403,
                name="Government Subsidy",
                description="All first sprints gain a subsidy bonus.",
                effect_type="adjust_specific_sprint_globally",
                target_sprint=1,
                delta_money=5000,
            ),
            IncidentCardConfig(
                card_id=404,
                name="Yellow Demand Boost",
                description="All future yellow sprints gain value.",
                effect_type="adjust_future_products",
                target_products=("yellow",),
                delta_money=5000,
            ),
            IncidentCardConfig(
                card_id=405,
                name="Black Product Breakthrough",
                description="The fourth black sprint becomes worth 100,000.",
                effect_type="set_specific_sprint_exact",
                target_products=("black",),
                target_sprint=4,
                set_value_money=100000,
            ),
        )

    first_product = normalized_names[0]
    last_product = normalized_names[-1]
    penalty_targets = tuple(normalized_names[1:3]) if len(normalized_names) > 2 else (last_product,)
    return (
        IncidentCardConfig(
            card_id=401,
            name="Demand Collapse",
            description="All future sprints of the first product become worth zero.",
            effect_type="set_future_product_to_zero",
            target_products=(first_product,),
        ),
        IncidentCardConfig(
            card_id=402,
            name="Competitive Pressure",
            description="Selected products lose value due to new competitors.",
            effect_type="adjust_future_products",
            target_products=penalty_targets,
            delta_money=-5000,
        ),
        IncidentCardConfig(
            card_id=403,
            name="Global Subsidy",
            description="All first sprints gain a subsidy bonus.",
            effect_type="adjust_specific_sprint_globally",
            target_sprint=1,
            delta_money=5000,
        ),
        IncidentCardConfig(
            card_id=404,
            name="Demand Boost",
            description="All future sprints of the first product gain value.",
            effect_type="adjust_future_products",
            target_products=(first_product,),
            delta_money=5000,
        ),
        IncidentCardConfig(
            card_id=405,
            name="Late Breakthrough",
            description="The last product's final sprint becomes especially valuable.",
            effect_type="set_specific_sprint_exact",
            target_products=(last_product,),
            target_sprint=4,
            set_value_money=100000,
        ),
    )


def _severity_multiplier_from_label(label: str) -> float:
    mapping = {
        "low": 0.75,
        "normal": 1.0,
        "high": 1.25,
    }
    return mapping.get(str(label).strip().lower(), 1.0)


def map_prototype_to_config(json_data: dict[str, Any]) -> GameConfig:
    """Translate the prototype export JSON into a canonical GameConfig."""
    basic = json_data.get("basic", {})
    products = json_data.get("products", [])
    layout = json_data.get("layout", {})
    refinements = json_data.get("refinements", {})
    incident = json_data.get("incident", {})

    product_names = [product.get("name", f"Product {index + 1}") for index, product in enumerate(products)]
    if not product_names:
        product_names = [
            f"Product {index + 1}"
            for index in range(int(basic.get("productsCount", 0)))
        ]

    raw_cells = layout.get("cells", [])
    board_ring_values = []
    board_features = []
    for row in raw_cells:
        board_ring_values.append([int(cell.get("value", 0)) for cell in row])
        board_features.append([int(cell.get("features", 1)) for cell in row])

    if not board_ring_values:
        products_count = int(basic.get("productsCount", len(product_names) or 1))
        sprints_per_product = int(basic.get("sprintsPerProduct", 1))
        board_ring_values = [[1 for _ in range(sprints_per_product)] for _ in range(products_count)]
        board_features = [[2 for _ in range(sprints_per_product)] for _ in range(products_count)]

    custom_cards = [
        IncidentCardConfig.from_dict(card_payload)
        for card_payload in incident.get("cards", [])
    ]
    if not custom_cards:
        custom_cards = list(build_default_incident_cards(product_names))

    sprint_count = len(board_ring_values[0]) if board_ring_values else int(basic.get("sprintsPerProduct", 1))
    custom_cards = [
        IncidentCardConfig(
            card_id=card.card_id,
            name=card.name,
            description=card.description,
            effect_type=card.effect_type,
            target_products=card.target_products,
            delta_money=card.delta_money,
            target_sprint=(
                None
                if card.target_sprint is None
                else max(1, min(int(card.target_sprint), sprint_count))
            ),
            set_value_money=card.set_value_money,
            future_only=card.future_only,
            weight=card.weight,
        )
        for card in custom_cards
    ]

    refinement_model = str(refinements.get("refinementModel", "Standard (ID 301)"))
    increase_range = (
        int(refinements.get("increaseRange", [refinements.get("refRollIncreaseMin", 1), refinements.get("refRollIncreaseMax", 2)])[0]),
        int(refinements.get("increaseRange", [refinements.get("refRollIncreaseMin", 1), refinements.get("refRollIncreaseMax", 2)])[1]),
    )
    decrease_range = (
        int(refinements.get("decreaseRange", [refinements.get("refRollDecreaseMin", 19), refinements.get("refRollDecreaseMax", 20)])[0]),
        int(refinements.get("decreaseRange", [refinements.get("refRollDecreaseMin", 19), refinements.get("refRollDecreaseMax", 20)])[1]),
    )

    return GameConfig.from_dict(
        {
            "schema_version": "1.0",
            "config_name": json_data.get("template", {}).get("name", basic.get("boardName", "Prototype Config")),
            "config_description": basic.get("boardDescription", ""),
            "players_count": int(basic.get("playersCount", 1)),
            "product_names": product_names,
            "max_turns": int(basic.get("roundsPerPlayer", basic.get("tokensPerPlayer", 6))),
            "starting_money": int(basic.get("startingMoney", 25000)),
            "ring_value": int(basic.get("costs", {}).get("ringValue", basic.get("ringValue", 5000))),
            "cost_continue": int(basic.get("costs", {}).get("costContinue", basic.get("costContinue", 0))),
            "cost_switch_mid": int(basic.get("costs", {}).get("costSwitchMid", basic.get("costSwitchMid", 5000))),
            "cost_switch_after": int(basic.get("costs", {}).get("costSwitchAfter", basic.get("costSwitchAfter", 0))),
            "mandatory_loan_amount": int(basic.get("costs", {}).get("mandatoryLoan", basic.get("mandatoryLoan", 50000))),
            "loan_interest": int(basic.get("costs", {}).get("loanInterest", basic.get("loanInterest", 5000))),
            "penalty_negative": int(basic.get("costs", {}).get("penaltyNeg", basic.get("penaltyNeg", 1000))),
            "penalty_positive": int(basic.get("costs", {}).get("penaltyPos", basic.get("penaltyPos", 1000))),
            "daily_scrums_per_sprint": int(
                basic.get("scrum", {}).get("dailyScrumsPerSprint", basic.get("dailyScrumsPerSprint", 5))
            ),
            "daily_scrum_target": int(
                basic.get("scrum", {}).get("dailyScrumTarget", basic.get("dailyScrumTarget", 12))
            ),
            "board_ring_values": board_ring_values,
            "board_features": board_features,
            "dice_rules": json_data.get(
                "dice_rules",
                [
                    {"min_features": 1, "max_features": 1, "dice_count": 1, "dice_sides": 20},
                    {"min_features": 2, "max_features": 2, "dice_count": 2, "dice_sides": 10},
                    {"min_features": 3, "max_features": None, "dice_count": 3, "dice_sides": 6},
                ],
            ),
            "refinement": {
                "active": str(refinements.get("refinementsActive", "Yes")).lower() == "yes",
                "model_name": refinement_model,
                "die_sides": int(refinements.get("dieSides", 20)),
                "product_rules": [
                    rule.to_dict()
                    for rule in build_default_refinement_rules(
                        product_names=product_names,
                        increase_range=increase_range,
                        decrease_range=decrease_range,
                        model_name=refinement_model,
                    )
                ],
            },
            "incident": {
                "active": str(incident.get("incidentsActive", "Yes")).lower() == "yes",
                "allow_player_specific_incidents": (
                    str(incident.get("allowPlayerSpecificIncidents", "No")).lower() == "yes"
                ),
                "draw_probability": float(incident.get("drawProbability", 1.0)),
                "severity_multiplier": _severity_multiplier_from_label(
                    incident.get("incidentFrequency", "Normal")
                ),
                "cards": [card.to_dict() for card in custom_cards],
            },
            "reserved_fields": {
                "board_id": json_data.get("boardId"),
                "template": json_data.get("template", {}),
                "tokens_per_player": basic.get("tokensPerPlayer"),
                "prototype_players": json_data.get("players", []),
                "prototype_products": json_data.get("products", []),
                "prototype_metadata": json_data.get("prototype", {}),
            },
        }
    )
