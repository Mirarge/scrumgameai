from __future__ import annotations

from pathlib import Path

from services.app_paths import (
    CURRENT_CHECKPOINT_DIR,
    DROPLET_RUNS_DIR,
    PLAYABLE_MODEL_V1_DIR,
    REFERENCE_V1_DIR,
    REPO_ROOT,
    RUNS_DIR,
)
from services.catalog_service import list_game_configs


def _engine_imports():
    from rl.checkpoint_utils import build_agent_for_config, load_checkpoint_payload  # noqa: E402
    from config.config_manager import compute_rule_signature, load_game_config  # noqa: E402
    return build_agent_for_config, load_checkpoint_payload, compute_rule_signature, load_game_config


def _checkpoint_id(checkpoint_path: Path) -> str:
    return checkpoint_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _checkpoint_type(checkpoint_path: Path) -> str:
    if checkpoint_path.name == "best_scrum_model.pth" or checkpoint_path.name.startswith("best_scrum_model"):
        return "best"
    if checkpoint_path.name == "latest_scrum_model.pth" or checkpoint_path.name.startswith("latest_scrum_model"):
        return "latest"
    return "intermediate"


def _source_label(source_type: str, source_run: str | None) -> str:
    if source_type == "run":
        return source_run or "run"
    if source_type == "current_artifacts":
        return "current artifacts"
    if source_type == "reference_v1":
        return "reference v1"
    if source_type == "playable_model_v1":
        return "playableModelV1"
    if source_type == "droplet_runs":
        return "droplet runs"
    return source_type


def _infer_shape_from_state_dict(state_dict) -> tuple[int | None, int | None]:
    first_weight = state_dict.get("network.0.weight")
    final_weight = state_dict.get("network.6.weight")
    state_dim = int(first_weight.shape[1]) if first_weight is not None and len(first_weight.shape) == 2 else None
    num_actions = int(final_weight.shape[0]) if final_weight is not None and len(final_weight.shape) == 2 else None
    return state_dim, num_actions


def _checkpoint_catalog_paths() -> list[tuple[Path, str, str | None]]:
    catalog = []

    if CURRENT_CHECKPOINT_DIR.exists():
        for checkpoint_path in sorted(CURRENT_CHECKPOINT_DIR.glob("*.pth")):
            if checkpoint_path.name.endswith(".policy.pth"):
                continue
            catalog.append((checkpoint_path, "current_artifacts", "current_artifacts"))

    if REFERENCE_V1_DIR.exists():
        for checkpoint_path in sorted(REFERENCE_V1_DIR.glob("*.pth")):
            if checkpoint_path.name.endswith(".policy.pth"):
                continue
            catalog.append((checkpoint_path, "reference_v1", None))

    if PLAYABLE_MODEL_V1_DIR.exists():
        for checkpoint_path in sorted(PLAYABLE_MODEL_V1_DIR.glob("*.pth")):
            if checkpoint_path.name.endswith(".policy.pth"):
                continue
            catalog.append((checkpoint_path, "playable_model_v1", None))

    if RUNS_DIR.exists():
        for run_dir in sorted((path for path in RUNS_DIR.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True):
            checkpoint_dir = run_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue
            for checkpoint_path in sorted(checkpoint_dir.glob("*.pth")):
                if checkpoint_path.name.endswith(".policy.pth"):
                    continue
                catalog.append((checkpoint_path, "run", run_dir.name))

    if DROPLET_RUNS_DIR.exists():
        for checkpoint_path in sorted(DROPLET_RUNS_DIR.glob("*.pth")):
            if checkpoint_path.name.endswith(".policy.pth"):
                continue
            catalog.append((checkpoint_path, "droplet_runs", None))

    return catalog


def _resolve_game_config_reference(game_config_id: str):
    _, _, _, load_game_config = _engine_imports()
    candidate_path = Path(game_config_id)
    if candidate_path.exists():
        return load_game_config(candidate_path)

    for item in list_game_configs():
        if item["id"] == game_config_id or item["path"] == game_config_id:
            return load_game_config(item["path"])

    raise ValueError(f"Game config `{game_config_id}` was not found.")
