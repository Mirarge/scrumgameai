import argparse
import csv
import random
import statistics
from pathlib import Path

import torch

from game_runtime.scrum_game_env import ScrumGameEnv
from rl.checkpoint_utils import load_agent_for_inference
from game.v2_deep_rl.rl.dqn_agile_agent import encode_state


BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "artifacts" / "runs"
DEFAULT_SEEDS = [42, 123, 999, 2026, 31415]


def parse_args():
    """Parse CLI arguments for the checkpoint-only robustness evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate one saved DDQN checkpoint across five greedy seeded games."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Optional path to a specific run folder. Defaults to the newest folder in artifacts/runs.",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir_flag",
        default=None,
        help="Optional path to a specific run folder. Overrides the positional run_dir argument.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seed list for pure greedy evaluation. The first five seeds are used.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Deprecated and ignored. This script does not train.",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=None,
        help="Deprecated and ignored. This script runs one full greedy game per seed.",
    )
    return parser.parse_args()


def resolve_input_path(path_text):
    """Resolve an input path relative to the current working directory or this script."""
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (BASE_DIR / candidate).resolve()


def get_newest_run_dir():
    """Return the most recently modified run folder under artifacts/runs."""
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Run directory does not exist: {RUNS_DIR}")

    run_dirs = [path for path in RUNS_DIR.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found under: {RUNS_DIR}")

    return max(run_dirs, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_run_dir(args):
    """Resolve either the requested run directory or the newest available one."""
    requested_path = args.run_dir_flag or args.run_dir
    if requested_path is None:
        run_dir = get_newest_run_dir()
    else:
        run_dir = resolve_input_path(requested_path)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run_dir}")

    return run_dir


def resolve_checkpoint_path(run_dir):
    """Return the expected best-model checkpoint path for one run folder."""
    checkpoint_path = run_dir / "checkpoints" / "best_scrum_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Expected best_scrum_model.pth inside the run's checkpoints folder."
        )
    return checkpoint_path


def load_agent(checkpoint_path):
    """Initialize the environment and load the saved DDQN policy for inference."""
    agent, env, metadata = load_agent_for_inference(checkpoint_path)
    return agent, metadata["resolved_game_config"]


def evaluate_one_seed(agent, game_config, seed):
    """Play one complete greedy game for one fixed seed."""
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
        "epsilon": 0.0,
        "episode_reward": total_reward,
        "ending_money": next_state["current_money"],
        "turns_played": steps,
        "loan_turns": env.turns_with_loan,
        "loans_taken": env.loans_taken,
        "invalid_action_count": invalid_action_count,
        "terminal_reason": info.get("terminal_reason", ""),
    }


def evaluate_across_seeds(agent, game_config, seeds):
    """Evaluate one saved policy across the requested fixed seeds."""
    return [evaluate_one_seed(agent, game_config, seed) for seed in seeds]


def save_results_csv(results, output_path):
    """Save the per-seed robustness results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_summary(results, run_dir, checkpoint_path, output_path):
    """Print a compact terminal summary for the evaluation batch."""
    rewards = [row["episode_reward"] for row in results]
    ending_money = [row["ending_money"] for row in results]
    bankruptcies = sum(1 for row in results if row["terminal_reason"] == "bankruptcy")

    print("DDQN Robustness Evaluation")
    print("-------------------------")
    print(f"Run folder: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Seeds: {[row['seed'] for row in results]}")
    print(f"Mean reward: {statistics.mean(rewards):.2f}")
    print(f"Mean ending money: {statistics.mean(ending_money):.2f}")
    print(f"Bankruptcies: {bankruptcies}/{len(results)}")
    print(f"Saved CSV to: {output_path}")


def main():
    """Run a pure inference robustness evaluation without any training calls."""
    args = parse_args()
    if len(args.seeds) < 5:
        raise ValueError("Provide at least five seeds for robustness evaluation.")

    run_dir = resolve_run_dir(args)
    checkpoint_path = resolve_checkpoint_path(run_dir)
    agent, game_config = load_agent(checkpoint_path)
    results = evaluate_across_seeds(agent, game_config, seeds=args.seeds[:5])
    output_path = run_dir / "robustness_results.csv"

    save_results_csv(results, output_path)
    print_summary(results, run_dir, checkpoint_path, output_path)


if __name__ == "__main__":
    main()
