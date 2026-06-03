from __future__ import annotations

import statistics
from pathlib import Path

from .catalog_service import list_game_configs
from .checkpoint_service import get_checkpoint_by_id
from .app_paths import ensure_engine_import_path

ensure_engine_import_path()

# torch-dependent engine imports are deferred so the API server starts without torch.


def _resolve_game_config(game_config_id: str):
    from config.config_manager import load_game_config  # noqa: E402
    candidate_path = Path(game_config_id)
    if candidate_path.exists():
        return load_game_config(candidate_path)

    from config.config_manager import load_game_config  # noqa: E402
    for item in list_game_configs():
        if item["id"] == game_config_id or item["path"] == game_config_id:
            return load_game_config(item["path"])
    raise ValueError(f"Game config `{game_config_id}` was not found.")


def _resolve_checkpoint(checkpoint_id: str) -> dict:
    checkpoint = get_checkpoint_by_id(checkpoint_id)
    if checkpoint is None:
        raise ValueError(f"Checkpoint `{checkpoint_id}` was not found.")
    return checkpoint


def _evaluate_one_seed(agent, game_config, seed: int) -> dict:
    import random  # noqa: E402
    import torch  # noqa: E402
    from game.v2_deep_rl.rl.dqn_agile_agent import encode_state  # noqa: E402
    from game_runtime.scrum_game_env import ScrumGameEnv  # noqa: E402
    random.seed(seed)
    torch.manual_seed(seed)

    env = ScrumGameEnv(game_config=game_config)
    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    total_reward = 0.0
    invalid_action_count = 0
    steps = 0

    while not done:
        action = agent.choose_action(state_vector, epsilon=0.0)
        next_state, reward, done, info = env.step(action)
        if info.get("invalid_action"):
            invalid_action_count += 1
        total_reward += reward
        steps += 1
        state_vector = encode_state(next_state, env)

    return {
        "seed": seed,
        "episode_reward": total_reward,
        "ending_money": next_state["current_money"],
        "turns_played": steps,
        "loan_turns": env.turns_with_loan,
        "loans_taken": env.loans_taken,
        "invalid_action_count": invalid_action_count,
        "terminal_reason": info.get("terminal_reason", ""),
    }


def _summarize_results(rows: list[dict]) -> dict:
    rewards = [row["episode_reward"] for row in rows]
    ending_money = [row["ending_money"] for row in rows]
    turns_played = [row["turns_played"] for row in rows]
    loans_taken = [row["loans_taken"] for row in rows]
    loan_turns = [row["loan_turns"] for row in rows]
    bankruptcies = sum(1 for row in rows if row["terminal_reason"] == "bankruptcy")
    invalid_actions = sum(row["invalid_action_count"] for row in rows)
    return {
        "seeds": [row["seed"] for row in rows],
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "stdev_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_ending_money": statistics.mean(ending_money) if ending_money else 0.0,
        "mean_turns_played": statistics.mean(turns_played) if turns_played else 0.0,
        "mean_loans_taken": statistics.mean(loans_taken) if loans_taken else 0.0,
        "mean_loan_turns": statistics.mean(loan_turns) if loan_turns else 0.0,
        "bankruptcies": bankruptcies,
        "invalid_actions": invalid_actions,
        "episodes": len(rows),
    }


def evaluate_checkpoint(payload: dict) -> dict:
    from rl.checkpoint_utils import load_agent_for_inference  # noqa: E402
    checkpoint = _resolve_checkpoint(payload["checkpoint_id"])
    game_config = _resolve_game_config(payload.get("game_config_id") or checkpoint["path"])
    seeds = payload.get("seeds") or [42]
    try:
        agent, _, _ = load_agent_for_inference(
            checkpoint["path"],
            game_config=game_config,
            strict_signature=False,
        )
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint `{checkpoint['label']}` cannot be evaluated on the selected config: {error}"
        ) from error
    rows = [_evaluate_one_seed(agent, game_config, int(seed)) for seed in seeds]
    return {
        "checkpoint": checkpoint,
        "summary": _summarize_results(rows),
        "results": rows,
    }


def compare_checkpoints(payload: dict) -> dict:
    left = evaluate_checkpoint(
        {
            "checkpoint_id": payload["left_checkpoint_id"],
            "game_config_id": payload["game_config_id"],
            "seeds": payload.get("seeds") or [42, 123, 999, 2026, 31415],
        }
    )
    right = evaluate_checkpoint(
        {
            "checkpoint_id": payload["right_checkpoint_id"],
            "game_config_id": payload["game_config_id"],
            "seeds": payload.get("seeds") or [42, 123, 999, 2026, 31415],
        }
    )
    return {
        "left": left,
        "right": right,
        "delta_mean_reward": left["summary"]["mean_reward"] - right["summary"]["mean_reward"],
        "delta_mean_ending_money": left["summary"]["mean_ending_money"] - right["summary"]["mean_ending_money"],
    }
