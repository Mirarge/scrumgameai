from __future__ import annotations

from pathlib import Path
import re

from services.app_paths import (
    CUSTOM_GAME_CONFIG_DIR,
    DEFAULT_GAME_CONFIG_PATH,
    DEFAULT_TRAINING_CONFIG_PATH,
    TRAINING_CONFIG_DIR,
    ensure_engine_import_path,
)

ensure_engine_import_path()

from config.config_manager import (  # noqa: E402
    GameConfig,
    TrainingConfig,
    compute_rule_signature,
    compute_training_signature,
    load_game_config,
    load_training_config,
    save_game_config,
    save_training_config,
    validate_game_config,
)


def _slugify_name(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_").lower()
    return slug or fallback


def _resolve_game_config_path(config_id_or_path: str) -> Path:
    candidate_path = Path(config_id_or_path)
    if candidate_path.exists():
        return candidate_path.resolve()

    if config_id_or_path == "default_game_config":
        return DEFAULT_GAME_CONFIG_PATH.resolve()

    for item in list_game_configs():
        if item["id"] == config_id_or_path:
            return Path(item["path"]).resolve()

    raise ValueError(f"Game config `{config_id_or_path}` was not found.")


def _resolve_training_config_path(config_id_or_path: str) -> Path:
    candidate_path = Path(config_id_or_path)
    if candidate_path.exists():
        return candidate_path.resolve()

    if config_id_or_path == "default_training_config":
        return DEFAULT_TRAINING_CONFIG_PATH.resolve()

    for item in list_training_configs():
        if item["id"] == config_id_or_path:
            return Path(item["path"]).resolve()

    raise ValueError(f"Training config `{config_id_or_path}` was not found.")


def list_game_configs() -> list[dict]:
    configs = []
    default_config = load_game_config(DEFAULT_GAME_CONFIG_PATH)
    configs.append(
        {
            "id": "default_game_config",
            "label": "Default Bundled Config",
            "path": str(DEFAULT_GAME_CONFIG_PATH),
            "rule_signature": compute_rule_signature(default_config),
            "products_count": default_config.products_count,
            "sprints_per_product": default_config.sprints_per_product,
            "config_name": default_config.config_name,
            "source": "bundled",
        }
    )

    if CUSTOM_GAME_CONFIG_DIR.exists():
        for config_path in sorted(CUSTOM_GAME_CONFIG_DIR.glob("*.json")):
            try:
                config = load_game_config(config_path)
            except (KeyError, TypeError, ValueError):
                continue
            configs.append(
                {
                    "id": config_path.stem,
                    "label": config_path.name,
                    "path": str(config_path),
                    "rule_signature": compute_rule_signature(config),
                    "products_count": config.products_count,
                    "sprints_per_product": config.sprints_per_product,
                    "config_name": config.config_name,
                    "source": "custom",
                }
            )
    return configs


def get_game_config(config_id_or_path: str) -> dict:
    config_path = _resolve_game_config_path(config_id_or_path)
    config = load_game_config(config_path)
    return {
        "id": "default_game_config" if config_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve() else config_path.stem,
        "label": config_path.name,
        "path": str(config_path),
        "source": "bundled" if config_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve() else "custom",
        "rule_signature": compute_rule_signature(config),
        "config": config.to_dict(),
    }


def list_training_configs() -> list[dict]:
    configs = []
    default_config = load_training_config(DEFAULT_TRAINING_CONFIG_PATH)
    configs.append(
        {
            "id": "default_training_config",
            "label": "Default Bundled Training Config",
            "path": str(DEFAULT_TRAINING_CONFIG_PATH),
            "training_signature": compute_training_signature(default_config),
            "episodes": default_config.episodes,
            "learning_rate": default_config.learning_rate,
            "gamma": default_config.gamma,
            "source": "bundled",
        }
    )

    if TRAINING_CONFIG_DIR.exists():
        for config_path in sorted(TRAINING_CONFIG_DIR.glob("*.json")):
            if config_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve():
                continue
            config = load_training_config(config_path)
            configs.append(
                {
                    "id": config_path.stem,
                    "label": config_path.name,
                    "path": str(config_path),
                    "training_signature": compute_training_signature(config),
                    "episodes": config.episodes,
                    "learning_rate": config.learning_rate,
                    "gamma": config.gamma,
                    "source": "custom",
                }
            )
    return configs


def get_training_config(config_id_or_path: str) -> dict:
    config_path = _resolve_training_config_path(config_id_or_path)
    config = load_training_config(config_path)
    return {
        "id": "default_training_config" if config_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve() else config_path.stem,
        "label": config_path.name,
        "path": str(config_path),
        "source": "bundled" if config_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve() else "custom",
        "training_signature": compute_training_signature(config),
        "config": config.to_dict(),
    }


def save_game_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Game config payload must include a `config` object.")

    game_config = GameConfig.from_dict(config_payload)
    target_id = payload.get("id")
    file_name = payload.get("file_name")

    if target_id:
        target_path = _resolve_game_config_path(target_id)
        if target_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve():
            raise ValueError("Bundled default game config cannot be overwritten.")
        if target_path.parent.resolve() != CUSTOM_GAME_CONFIG_DIR.resolve():
            raise ValueError("Only managed custom game configs can be overwritten.")
    else:
        file_stub = _slugify_name(file_name or game_config.config_name, "game_config")
        target_path = CUSTOM_GAME_CONFIG_DIR / f"{file_stub}.json"

    CUSTOM_GAME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    save_game_config(game_config, target_path)
    return get_game_config(str(target_path))


def delete_game_config_asset(config_id_or_path: str) -> dict:
    target_path = _resolve_game_config_path(config_id_or_path)
    if target_path.resolve() == DEFAULT_GAME_CONFIG_PATH.resolve():
        raise ValueError("Bundled default game config cannot be deleted.")
    if target_path.parent.resolve() != CUSTOM_GAME_CONFIG_DIR.resolve():
        raise ValueError("Only managed custom game configs can be deleted.")
    if not target_path.exists():
        raise ValueError(f"Game config `{config_id_or_path}` was not found.")
    target_path.unlink()
    return {"deleted": True, "id": target_path.stem, "path": str(target_path)}


def validate_game_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Game config payload must include a `config` object.")
    game_config = GameConfig.from_dict(config_payload)
    validate_game_config(game_config)
    return {
        "valid": True,
        "rule_signature": compute_rule_signature(game_config),
        "products_count": game_config.products_count,
        "sprints_per_product": game_config.sprints_per_product,
        "actions_count": game_config.products_count + 1,
        "config_name": game_config.config_name,
    }


def save_training_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Training config payload must include a `config` object.")

    training_config = TrainingConfig.from_dict(config_payload)
    target_id = payload.get("id")
    file_name = payload.get("file_name")

    if target_id:
        target_path = _resolve_training_config_path(target_id)
        if target_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve():
            raise ValueError("Bundled default training config cannot be overwritten.")
        if target_path.parent.resolve() != TRAINING_CONFIG_DIR.resolve():
            raise ValueError("Only managed custom training configs can be overwritten.")
    else:
        file_stub = _slugify_name(file_name or "training_config", "training_config")
        target_path = TRAINING_CONFIG_DIR / f"{file_stub}.json"

    TRAINING_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    save_training_config(training_config, target_path)
    return get_training_config(str(target_path))


def delete_training_config_asset(config_id_or_path: str) -> dict:
    target_path = _resolve_training_config_path(config_id_or_path)
    if target_path.resolve() == DEFAULT_TRAINING_CONFIG_PATH.resolve():
        raise ValueError("Bundled default training config cannot be deleted.")
    if target_path.parent.resolve() != TRAINING_CONFIG_DIR.resolve():
        raise ValueError("Only managed custom training configs can be deleted.")
    if not target_path.exists():
        raise ValueError(f"Training config `{config_id_or_path}` was not found.")
    target_path.unlink()
    return {"deleted": True, "id": target_path.stem, "path": str(target_path)}


def validate_training_config_asset(payload: dict) -> dict:
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("Training config payload must include a `config` object.")
    training_config = TrainingConfig.from_dict(config_payload)
    return {
        "valid": True,
        "training_signature": compute_training_signature(training_config),
        "episodes": training_config.episodes,
        "learning_rate": training_config.learning_rate,
        "gamma": training_config.gamma,
        "batch_size": training_config.batch_size,
    }
