import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from rl.checkpoint_utils import (
    build_agent_for_config,
    load_agent_from_checkpoint,
    load_checkpoint_payload,
    save_checkpoint,
)
from config.config_manager import (
    GameConfig,
    TrainingConfig,
    compute_rule_signature,
    compute_training_signature,
    load_game_config,
    load_training_config,
    save_game_config,
    save_training_config,
)
from game_rules.rule_randomization import sample_game_config
from game_runtime.scrum_game_env import ScrumGameEnv
from rl.dqn_waterfall_agent import encode_state
from rl.model_utils import save_metrics_json
from play.match_runner import valid_actions_for_state


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

def print_progress_bar(current_episode, total_episodes, bar_length=50):
    """Print a progress bar for training episodes."""
    percent = current_episode / total_episodes
    filled = int(bar_length * percent)
    try:
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\rProgress: [{bar}] {percent*100:.1f}% ({current_episode}/{total_episodes})", flush=True)
    except:
        bar = '[]' * filled + '{}' * (bar_length - filled)
        print(f"\rProgress: [{bar}] {percent*100:.1f}% ({current_episode}/{total_episodes})", flush=True)


def _slugify_run_name(value: str | None) -> str:
    """Convert an optional user-facing run name into a safe folder suffix."""
    text = str(value or "").strip().lower()
    slug = "".join(character if character.isalnum() else "_" for character in text)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:48]


def ensure_deep_rl_directories():
    """Create the deep-RL artifact folders used by training."""
    checkpoint_dir = BASE_DIR / "artifacts" / "checkpoints"
    plot_dir = BASE_DIR / "artifacts" / "plots"
    report_dir = BASE_DIR / "artifacts" / "reports"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, plot_dir, report_dir


def create_timestamped_run_directory(run_name: str | None = None):
    """Create a unique timestamped run directory under artifacts/runs."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = datetime.now().strftime("run_%Y-%m-%d_%H%M")
    run_suffix = _slugify_run_name(run_name)
    if run_suffix:
        base_name = f"{base_name}_{run_suffix}"
    candidate = RUNS_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = RUNS_DIR / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def resolve_output_directories(run_dir=None):
    """Resolve either the legacy flat artifact directories or a timestamped run folder."""
    if run_dir is None:
        checkpoint_dir, plot_dir, report_dir = ensure_deep_rl_directories()
        return checkpoint_dir, plot_dir, report_dir, None, ARTIFACTS_DIR

    run_path = Path(run_dir)
    if not run_path.is_absolute():
        run_path = BASE_DIR / run_path

    checkpoint_dir = run_path / "checkpoints"
    plot_dir = run_path / "plots"
    report_dir = run_path / "reports"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, plot_dir, report_dir, run_path, run_path


def initialize_training_log(log_path, num_actions):
    """Create or overwrite the live DQN training log CSV."""
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "epsilon",
                "episode_reward",
                "rolling_average_reward",
                "mean_recent_loss",
                "replay_buffer_size",
                "average_loan_duration",
                "bankruptcy_count",
                "average_ending_money",
                "invalid_action_count",
            ]
            + [f"action_{action_id}_count" for action_id in range(num_actions)]
        )


def initialize_evaluation_log(log_path):
    """Create or overwrite the periodic evaluation CSV."""
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "average_reward",
                "bankruptcy_rate",
                "average_ending_money",
                "average_loan_duration",
                "invalid_action_rate",
            ]
        )


def append_training_log(
    log_path,
    episode,
    epsilon,
    episode_reward,
    rolling_average_reward,
    mean_recent_loss,
    replay_buffer_size,
    average_loan_duration,
    bankruptcy_count,
    average_ending_money,
    invalid_action_count,
    action_counts,
):
    """Append one summarized training record for the dashboard."""
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                episode,
                epsilon,
                episode_reward,
                rolling_average_reward,
                mean_recent_loss,
                replay_buffer_size,
                average_loan_duration,
                bankruptcy_count,
                average_ending_money,
                invalid_action_count,
            ]
            + list(action_counts)
        )


def append_evaluation_log(log_path, episode, metrics):
    """Append one periodic evaluation record."""
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                episode,
                metrics["average_reward"],
                metrics["bankruptcy_rate"],
                metrics["average_ending_money"],
                metrics["average_loan_duration"],
                metrics["invalid_action_rate"],
            ]
        )


def rolling_average(values, window_size=500):
    """Compute a rolling mean for the DDQN learning curve."""
    if not values:
        return []

    smoothed = []
    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def epsilon_by_episode(
    episode,
    episodes,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay_end=0.9
):
    """
    Linearly decay epsilon very slowly across the first 45,000 episodes.

    The 8-action branch needs more exploration than the earlier binary-action setup.
    """
    if episode >= episodes*epsilon_decay_end:
        return epsilon_min
    

    progress = episode / float(episodes)
    return epsilon_start - (epsilon_start - epsilon_min) * progress


def resolve_training_config(
    training_config: TrainingConfig | None = None,
    training_config_path=None,
    num_episodes=None,
    learning_rate=None,
    gamma=None,
    checkpoint_interval=None,
    evaluation_interval=None,
    evaluation_episodes=None,
    seed=None,
    epsilon_decay_end=None,
    rule_randomization_enabled=None,
    rule_randomization_frequency=None,
    rule_randomization_eval_configs=None,
    rule_randomization_bounds=None,
    auto_continue_enabled=None,
    auto_continue_cycles=None,
    run_notes="",
):
    """Load the base training config and apply CLI/function overrides."""
    base_config = training_config or load_training_config(training_config_path)
    payload = base_config.to_dict()

    if num_episodes is not None:
        payload["episodes"] = int(num_episodes)
    if learning_rate is not None:
        payload["learning_rate"] = float(learning_rate)
    if gamma is not None:
        payload["gamma"] = float(gamma)
    if checkpoint_interval is not None:
        payload["checkpoint_interval"] = int(checkpoint_interval)
    if evaluation_interval is not None:
        payload["evaluation_interval"] = int(evaluation_interval)
    if evaluation_episodes is not None:
        payload["evaluation_episodes"] = int(evaluation_episodes)
    if seed is not None:
        payload["seed"] = int(seed)
    if epsilon_decay_end is not None:
        payload["epsilon_decay_end"] = int(epsilon_decay_end)
    if rule_randomization_enabled is not None:
        payload["rule_randomization_enabled"] = bool(rule_randomization_enabled)
    if rule_randomization_frequency is not None:
        payload["rule_randomization_frequency"] = int(rule_randomization_frequency)
    if rule_randomization_eval_configs is not None:
        payload["rule_randomization_eval_configs"] = int(rule_randomization_eval_configs)
    if rule_randomization_bounds is not None:
        payload["rule_randomization_bounds"] = dict(rule_randomization_bounds)
    if auto_continue_enabled is not None:
        payload["auto_continue_enabled"] = bool(auto_continue_enabled)
    if auto_continue_cycles is not None:
        payload["auto_continue_cycles"] = int(auto_continue_cycles)
    if run_notes:
        payload["run_notes"] = str(run_notes)

    return TrainingConfig.from_dict(payload)


def initialize_agent_from_checkpoint(
    agent,
    checkpoint_path,
    target_game_config: GameConfig,
    resume_mode: str,
):
    """Warm-start an agent from an existing checkpoint under the requested resume mode."""
    if not checkpoint_path:
        return None

    if resume_mode not in {"strict", "fine-tune"}:
        raise ValueError("resume_mode must be either 'strict' or 'fine-tune'.")

    checkpoint_path = Path(checkpoint_path)
    strict_signature = resume_mode == "strict"
    try:
        _, _, checkpoint_metadata = load_agent_from_checkpoint(
            checkpoint_path,
            game_config=target_game_config,
            strict_signature=strict_signature,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Could not resume from checkpoint `{checkpoint_path}` under mode `{resume_mode}`: {error}"
        ) from error

    payload = load_checkpoint_payload(checkpoint_path, map_location=agent.device)
    state_dict = payload["model_state_dict"]

    try:
        agent.policy_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(payload.get("target_model_state_dict", state_dict))
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint tensor shapes do not match the requested game config. "
            "Resume/fine-tune only works when the model architecture is still compatible."
        ) from error

    if resume_mode == "strict":
        try:
            agent.load_training_state_dict(payload, include_replay=True)
        except (RuntimeError, ValueError, KeyError) as error:
            raise RuntimeError(
                "Checkpoint training state could not be restored for strict resume. "
                "Use fine-tune mode to load only compatible model weights."
            ) from error

    agent.policy_network.train()
    agent.target_network.eval()
    checkpoint_episode = checkpoint_metadata.get("episode")
    if checkpoint_episode is None:
        checkpoint_episode = 0
    checkpoint_average_reward = checkpoint_metadata.get("average_reward")
    return {
        "resume_checkpoint_path": str(checkpoint_path),
        "resume_mode": resume_mode,
        "resume_checkpoint_episode": int(checkpoint_episode),
        "resume_checkpoint_average_reward": checkpoint_average_reward,
        "resume_restored_optimizer": bool(resume_mode == "strict" and payload.get("optimizer_state_dict") is not None),
        "resume_restored_replay_buffer": bool(resume_mode == "strict" and payload.get("replay_buffer") is not None),
        "resume_replay_buffer_size": len(agent.replay_buffer),
        "resume_training_steps": agent.training_steps,
        "resume_checkpoint_rule_signature": (
            checkpoint_metadata.get("checkpoint_rule_signature")
            or checkpoint_metadata.get("rule_signature")
        ),
        "resume_target_rule_signature": compute_rule_signature(target_game_config),
        "resume_legacy_checkpoint": checkpoint_metadata.get("legacy_checkpoint", False),
    }


def evaluate_dqn_agent(agent, num_episodes=1000, seed=1042, game_config: GameConfig | None = None):
    """
    Evaluate the DDQN greedily with epsilon fixed at 0.

    Evaluation uses a separate environment instance and no optimization steps.
    Global random state is saved and restored so interleaved training is unaffected.
    """
    saved_random_state = random.getstate()
    saved_torch_state = torch.get_rng_state()

    random.seed(seed)
    torch.manual_seed(seed)

    try:
        env = ScrumGameEnv(game_config=game_config)
        evaluation_rewards = []
        ending_monies = []
        bankruptcy_count = 0
        loan_durations = []
        action_counts = [0] * env.num_actions
        invalid_action_count = 0

        for episode in range(num_episodes):
            state = env.reset(seed=seed + episode)
            state_vector = encode_state(state, env)
            done = False
            cumulative_reward = 0

            while not done:
                valid_actions = valid_actions_for_state(env, state)
                action = agent.choose_action(state_vector, epsilon=0.0, valid_actions=valid_actions)
                action_counts[action] += 1
                next_state, reward, done, info = env.step(action)
                next_state_vector = encode_state(next_state, env)
                if info.get("invalid_action"):
                    invalid_action_count += 1

                cumulative_reward += reward
                state_vector = next_state_vector

                if done and info.get("terminal_reason") == "bankruptcy":
                    bankruptcy_count += 1
                    ending_monies.append(next_state["current_money"])
                    loan_durations.append(env.turns_with_loan)
                elif done:
                    ending_monies.append(next_state["current_money"])
                    loan_durations.append(env.turns_with_loan)

            evaluation_rewards.append(cumulative_reward)

        return {
            "rewards": evaluation_rewards,
            "average_reward": sum(evaluation_rewards) / len(evaluation_rewards),
            "average_ending_money": sum(ending_monies) / len(ending_monies),
            "bankruptcy_rate": bankruptcy_count / len(evaluation_rewards),
            "average_loan_duration": sum(loan_durations) / len(loan_durations),
            "invalid_action_rate": invalid_action_count / max(sum(action_counts), 1),
            "action_counts": action_counts,
        }
    finally:
        random.setstate(saved_random_state)
        torch.set_rng_state(saved_torch_state)


def evaluate_dqn_agent_on_randomized_rules(
    agent,
    base_game_config: GameConfig,
    training_config: TrainingConfig,
    num_episodes=1000,
    seed=1042,
):
    """Evaluate greedily across held-out sampled rule configs."""
    config_count = max(1, int(training_config.rule_randomization_eval_configs))
    episodes_per_config = max(1, int(num_episodes) // config_count)
    rng = random.Random(seed)
    per_config_metrics = []
    all_rewards = []
    all_action_counts = None
    bankruptcy_total = 0.0
    ending_money_total = 0.0
    loan_duration_total = 0.0
    invalid_action_total = 0.0

    for config_index in range(config_count):
        sampled_config = sample_game_config(
            base_game_config,
            rng,
            bounds=training_config.rule_randomization_bounds,
            config_name=f"heldout_eval_{config_index + 1}",
        )
        metrics = evaluate_dqn_agent(
            agent,
            num_episodes=episodes_per_config,
            seed=seed + config_index * 10000,
            game_config=sampled_config,
        )
        per_config_metrics.append(
            {
                "config_index": config_index + 1,
                "rule_signature": compute_rule_signature(sampled_config),
                "average_reward": metrics["average_reward"],
                "bankruptcy_rate": metrics["bankruptcy_rate"],
                "average_ending_money": metrics["average_ending_money"],
                "invalid_action_rate": metrics["invalid_action_rate"],
            }
        )
        all_rewards.extend(metrics["rewards"])
        if all_action_counts is None:
            all_action_counts = [0] * len(metrics["action_counts"])
        for action_id, count in enumerate(metrics["action_counts"]):
            all_action_counts[action_id] += count
        bankruptcy_total += metrics["bankruptcy_rate"] * len(metrics["rewards"])
        ending_money_total += metrics["average_ending_money"] * len(metrics["rewards"])
        loan_duration_total += metrics["average_loan_duration"] * len(metrics["rewards"])
        invalid_action_total += metrics["invalid_action_rate"] * max(sum(metrics["action_counts"]), 1)

    total_episodes = max(len(all_rewards), 1)
    total_actions = max(sum(all_action_counts or [0]), 1)
    sorted_config_rewards = sorted(item["average_reward"] for item in per_config_metrics)
    worst_count = max(1, int(len(sorted_config_rewards) * 0.1))

    return {
        "rewards": all_rewards,
        "average_reward": sum(all_rewards) / total_episodes,
        "average_ending_money": ending_money_total / total_episodes,
        "bankruptcy_rate": bankruptcy_total / total_episodes,
        "average_loan_duration": loan_duration_total / total_episodes,
        "invalid_action_rate": invalid_action_total / total_actions,
        "action_counts": all_action_counts or [],
        "randomized_rule_evaluation": True,
        "configs_evaluated": config_count,
        "episodes_per_config": episodes_per_config,
        "worst_config_average_reward": sorted_config_rewards[0],
        "worst_10pct_average_reward": sum(sorted_config_rewards[:worst_count]) / worst_count,
        "per_config_metrics": per_config_metrics,
    }


def evaluate_for_training_config(
    agent,
    game_config: GameConfig,
    training_config: TrainingConfig,
    num_episodes=1000,
    seed=1042,
):
    if training_config.rule_randomization_enabled:
        return evaluate_dqn_agent_on_randomized_rules(
            agent,
            game_config,
            training_config,
            num_episodes=num_episodes,
            seed=seed,
        )
    return evaluate_dqn_agent(
        agent,
        num_episodes=num_episodes,
        seed=seed,
        game_config=game_config,
    )


def save_training_plot(training_rewards, output_path):
    """Save the DDQN rolling-average training curve."""
    smoothed_rewards = rolling_average(training_rewards, window_size=500)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label="Rolling Average Reward (500 episodes)", linewidth=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward")
    plt.title("Double DQN Training Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_dqn_agent(
    num_episodes=500000,
    learning_rate=0.0005,
    gamma=None,
    checkpoint_interval=10000,
    evaluation_interval=10000,
    evaluation_episodes=100,
    seed=42,
    epsilon_decay_end=None,
    rule_randomization_enabled=None,
    rule_randomization_frequency=None,
    rule_randomization_eval_configs=None,
    rule_randomization_bounds=None,
    auto_continue_enabled=None,
    auto_continue_cycles=None,
    run_dir=None,
    run_notes="",
    game_config: GameConfig | None = None,
    game_config_path=None,
    training_config: TrainingConfig | None = None,
    training_config_path=None,
    resume_from=None,
    resume_mode="strict",
    resume_episodes_mode="incremental",
    agentType = "agile",
):
    """Train a Double DQN agent on the advanced Scrum Game environment."""
    if resume_episodes_mode not in {"incremental", "absolute"}:
        raise ValueError("resume_episodes_mode must be either 'incremental' or 'absolute'.")

    resolved_game_config = game_config or load_game_config(game_config_path)
    resolved_training_config = resolve_training_config(
        training_config=training_config,
        training_config_path=training_config_path,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        gamma=gamma,
        checkpoint_interval=checkpoint_interval,
        evaluation_interval=evaluation_interval,
        evaluation_episodes=evaluation_episodes,
        seed=seed,
        epsilon_decay_end=epsilon_decay_end,
        rule_randomization_enabled=rule_randomization_enabled,
        rule_randomization_frequency=rule_randomization_frequency,
        rule_randomization_eval_configs=rule_randomization_eval_configs,
        rule_randomization_bounds=rule_randomization_bounds,
        auto_continue_enabled=auto_continue_enabled,
        auto_continue_cycles=auto_continue_cycles,
        run_notes=run_notes,
    )

    random.seed(resolved_training_config.seed)
    torch.manual_seed(resolved_training_config.seed)

    checkpoint_dir, plot_dir, report_dir, run_path, config_root = resolve_output_directories(run_dir)

    env = ScrumGameEnv(game_config=resolved_game_config)
    initial_state = env.reset(seed=resolved_training_config.seed)
    state_dim = len(encode_state(initial_state, env))
    num_actions = env.num_actions
    randomization_rng = random.Random(resolved_training_config.seed + 97117)
    active_training_env = env

    log_path = report_dir / "logs.csv"
    evaluation_log_path = report_dir / "evaluation_history.csv"
    initialize_training_log(log_path, num_actions)
    initialize_evaluation_log(evaluation_log_path)

    game_config_output_path = config_root / "game_config.json"
    training_config_output_path = config_root / "training_config.json"
    save_game_config(resolved_game_config, game_config_output_path)
    save_training_config(resolved_training_config, training_config_output_path)

    run_metadata = {
        "run_name": run_path.name if run_path is not None else "current_artifacts",
        "run_path": str(run_path) if run_path is not None else str(config_root),
        "created_at": datetime.now().isoformat(),
        "training_episodes": resolved_training_config.episodes,
        "learning_rate": resolved_training_config.learning_rate,
        "gamma": resolved_training_config.gamma,
        "checkpoint_interval": resolved_training_config.checkpoint_interval,
        "evaluation_interval": resolved_training_config.evaluation_interval,
        "evaluation_episodes": resolved_training_config.evaluation_episodes,
        "seed": resolved_training_config.seed,
        "run_notes": resolved_training_config.run_notes,
        "game_config_path": str(game_config_output_path),
        "training_config_path": str(training_config_output_path),
        "rule_signature": compute_rule_signature(resolved_game_config),
        "training_signature": compute_training_signature(resolved_training_config),
        "resume_checkpoint_path": str(resume_from) if resume_from else None,
        "resume_mode": resume_mode if resume_from else None,
        "resume_episodes_mode": resume_episodes_mode if resume_from else None,
        "rule_randomization_enabled": resolved_training_config.rule_randomization_enabled,
        "rule_randomization_frequency": resolved_training_config.rule_randomization_frequency,
        "rule_randomization_eval_configs": resolved_training_config.rule_randomization_eval_configs,
        "auto_continue_enabled": resolved_training_config.auto_continue_enabled,
        "auto_continue_cycles": resolved_training_config.auto_continue_cycles,
    }

    if run_path is not None:
        save_metrics_json(run_metadata, str(run_path / "run_metadata.json"))
    else:
        save_metrics_json(run_metadata, str(report_dir / "run_metadata.json"))

    agent, _ = build_agent_for_config(
        resolved_game_config,
        learning_rate=resolved_training_config.learning_rate,
        gamma=resolved_training_config.gamma,
        replay_capacity=resolved_training_config.replay_capacity,
        batch_size=resolved_training_config.batch_size,
        target_update_frequency=resolved_training_config.target_update_frequency,
        agentType=agentType,
    )
    resume_metadata = initialize_agent_from_checkpoint(
        agent,
        resume_from,
        resolved_game_config,
        resume_mode,
    )
    resume_start_episode = int((resume_metadata or {}).get("resume_checkpoint_episode") or 0)
    first_episode = resume_start_episode + 1
    configured_episodes = int(resolved_training_config.episodes)
    if resume_metadata and resume_episodes_mode == "absolute":
        final_episode = configured_episodes
    else:
        final_episode = resume_start_episode + configured_episodes
    episodes_this_run = max(0, final_episode - resume_start_episode)
    best_average_reward = float("-inf")
    if resume_metadata and resume_metadata.get("resume_checkpoint_average_reward") is not None:
        best_average_reward = float(resume_metadata["resume_checkpoint_average_reward"])

    run_metadata.update(
        {
            "episodes_this_run": episodes_this_run,
            "start_episode": first_episode,
            "end_episode": final_episode,
            "resume_start_episode": resume_start_episode if resume_metadata else None,
        }
    )
    if resume_metadata:
        run_metadata.update(resume_metadata)
    if run_path is not None:
        save_metrics_json(run_metadata, str(run_path / "run_metadata.json"))
    else:
        save_metrics_json(run_metadata, str(report_dir / "run_metadata.json"))

    training_rewards = []
    training_losses = []
    episode_loan_durations = []
    bankruptcy_flags = []
    ending_monies = []
    recent_action_counts = []
    invalid_action_flags = []

    best_checkpoint_path = checkpoint_dir / "best_scrum_model.pth"
    evaluations_completed = 0
    if resume_metadata:
        save_checkpoint(
            best_checkpoint_path,
            agent,
            resolved_game_config,
            resolved_training_config,
            extra_metadata={
                "episode": resume_start_episode,
                "average_reward": (
                    best_average_reward if best_average_reward != float("-inf") else None
                ),
                "is_best_checkpoint": True,
                **resume_metadata,
            },
        )

    # training loop
    for episode in range(first_episode, final_episode + 1):
        block_episode = episode - resume_start_episode
        if resolved_training_config.rule_randomization_enabled and (
            block_episode == 1
            or (block_episode - 1) % resolved_training_config.rule_randomization_frequency == 0
        ):
            sampled_game_config = sample_game_config(
                resolved_game_config,
                randomization_rng,
                bounds=resolved_training_config.rule_randomization_bounds,
                config_name=f"train_episode_{episode}",
            )
            active_training_env = ScrumGameEnv(game_config=sampled_game_config)

        state = active_training_env.reset(seed=resolved_training_config.seed + episode)
        state_vector = encode_state(state, active_training_env)
        done = False
        cumulative_reward = 0
        bankruptcy_this_episode = 0
        invalid_actions_this_episode = 0
        episode_action_counts = [0] * num_actions

        epsilon = epsilon_by_episode(
            episode - 1,
            num_episodes,
            epsilon_start=resolved_training_config.epsilon_start,
            epsilon_min=resolved_training_config.epsilon_min,
            epsilon_decay_end=resolved_training_config.epsilon_decay_end,
        )

        # Clear screen using ANSI escape codes
        print("\033[H\033[2J", end="")
        print_progress_bar(episode, final_episode)
        while not done:
            valid_actions = valid_actions_for_state(active_training_env, state)
            action = agent.choose_action(state_vector, epsilon=epsilon, valid_actions=valid_actions)
            episode_action_counts[action] += 1

            next_state, reward, done, info = active_training_env.step(action)
            next_state_vector = encode_state(next_state, active_training_env)
            if info.get("invalid_action"):
                invalid_actions_this_episode += 1

            agent.store_transition(state_vector, action, reward, next_state_vector, done)
            loss = agent.train_step()

            if loss is not None:
                training_losses.append(loss)

            cumulative_reward += reward
            state_vector = next_state_vector
            state = next_state

            if done and info.get("terminal_reason") == "bankruptcy":
                bankruptcy_this_episode = 1

        training_rewards.append(cumulative_reward)
        episode_loan_durations.append(active_training_env.turns_with_loan)
        bankruptcy_flags.append(bankruptcy_this_episode)
        ending_monies.append(active_training_env.current_money)
        recent_action_counts.append(episode_action_counts)
        invalid_action_flags.append(invalid_actions_this_episode)

        if block_episode % 100 == 0:
            recent_rewards = training_rewards[-100:]
            recent_losses = training_losses[-100:] if training_losses else []
            recent_loan_durations = episode_loan_durations[-100:]
            recent_bankruptcies = bankruptcy_flags[-100:]
            recent_ending_monies = ending_monies[-100:]
            recent_invalid_actions = invalid_action_flags[-100:]
            block_action_counts = [0] * num_actions
            for action_counts in recent_action_counts[-100:]:
                for action_id, count in enumerate(action_counts):
                    block_action_counts[action_id] += count

            append_training_log(
                log_path=log_path,
                episode=episode,
                epsilon=epsilon,
                episode_reward=cumulative_reward,
                rolling_average_reward=sum(recent_rewards) / len(recent_rewards),
                mean_recent_loss=(sum(recent_losses) / len(recent_losses)) if recent_losses else 0.0,
                replay_buffer_size=len(agent.replay_buffer),
                average_loan_duration=(sum(recent_loan_durations) / len(recent_loan_durations)) if recent_loan_durations else 0.0,
                bankruptcy_count=sum(recent_bankruptcies),
                average_ending_money=(sum(recent_ending_monies) / len(recent_ending_monies)) if recent_ending_monies else 0.0,
                invalid_action_count=sum(recent_invalid_actions),
                action_counts=block_action_counts,
            )

        if block_episode % resolved_training_config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode:06d}.pth"
            try:
                save_checkpoint(
                    checkpoint_path,
                    agent,
                    resolved_game_config,
                    resolved_training_config,
                    extra_metadata={
                        "episode": episode,
                        **(resume_metadata or {}),
                    },
                )
                print(f"Checkpoint saved at episode {episode}: {checkpoint_path}")
            except Exception as _ckpt_err:
                # Non-fatal — OneDrive/antivirus can lock .pth files on Windows.
                # Log the failure and continue training rather than crashing the run.
                print(f"[WARNING] Checkpoint save failed at episode {episode} (skipping): {_ckpt_err}")

        if block_episode % resolved_training_config.evaluation_interval == 0:
            evaluation_metrics = evaluate_for_training_config(
                agent,
                resolved_game_config,
                resolved_training_config,
                num_episodes=resolved_training_config.evaluation_episodes,
                seed=resolved_training_config.seed + 100000 + episode,
            )
            append_evaluation_log(evaluation_log_path, episode, evaluation_metrics)
            evaluations_completed += 1

            if evaluation_metrics["average_reward"] > best_average_reward:
                best_average_reward = evaluation_metrics["average_reward"]
                try:
                    save_checkpoint(
                        best_checkpoint_path,
                        agent,
                        resolved_game_config,
                        resolved_training_config,
                        extra_metadata={
                            "episode": episode,
                            "average_reward": best_average_reward,
                            "is_best_checkpoint": True,
                            **(resume_metadata or {}),
                        },
                    )
                    print(
                        f"Updated best model at episode {episode}: {best_checkpoint_path} "
                        f"(avg reward {best_average_reward:.2f})"
                    )
                except Exception as _ckpt_err:
                    print(f"[WARNING] Best-model save failed at episode {episode} (skipping): {_ckpt_err}")

    if not best_checkpoint_path.exists() or evaluations_completed == 0:
        try:
            save_checkpoint(
                best_checkpoint_path,
                agent,
                resolved_game_config,
                resolved_training_config,
                extra_metadata={
                    "episode": final_episode,
                    "is_best_checkpoint": True,
                    **(resume_metadata or {}),
                },
            )
        except Exception as _ckpt_err:
            print(f"[WARNING] Final best-model save failed: {_ckpt_err}")

    # Always save the final state so autopilot can chain continuations from the
    # true last episode rather than from the (potentially much earlier) best-reward
    # episode stored in best_scrum_model.pth.
    latest_checkpoint_path = checkpoint_dir / "latest_scrum_model.pth"
    try:
        save_checkpoint(
            latest_checkpoint_path,
            agent,
            resolved_game_config,
            resolved_training_config,
            extra_metadata={
                "episode": final_episode,
                "average_reward": best_average_reward if best_average_reward != float("-inf") else None,
                **(resume_metadata or {}),
            },
        )
    except Exception as _ckpt_err:
        print(f"[WARNING] Final latest-model save failed: {_ckpt_err}")

    plot_path = plot_dir / "dqn_training_curve.png"
    save_training_plot(training_rewards, output_path=plot_path)

    try:
        best_agent, _, _ = load_agent_from_checkpoint(
            best_checkpoint_path,
            game_config=resolved_game_config,
        )
    except Exception as _load_err:
        print(f"[WARNING] Could not reload best checkpoint for final evaluation ({_load_err}); using final training agent.")
        best_agent = agent

    final_evaluation = evaluate_for_training_config(
        best_agent,
        resolved_game_config,
        resolved_training_config,
        num_episodes=1000,
        seed=resolved_training_config.seed + 1000,
    )

    save_metrics_json(
        {
            "model": "Double DQN",
            "training_episodes": resolved_training_config.episodes,
            "episodes_this_run": episodes_this_run,
            "start_episode": first_episode,
            "end_episode": final_episode,
            "resume_start_episode": resume_start_episode if resume_metadata else None,
            "evaluation_episodes": len(final_evaluation["rewards"]),
            "average_reward_per_episode": final_evaluation["average_reward"],
            "average_ending_money": final_evaluation["average_ending_money"],
            "bankruptcy_rate": final_evaluation["bankruptcy_rate"],
            "average_loan_duration": final_evaluation["average_loan_duration"],
            "invalid_action_rate": final_evaluation["invalid_action_rate"],
            "randomized_rule_evaluation": final_evaluation.get("randomized_rule_evaluation", False),
            "configs_evaluated": final_evaluation.get("configs_evaluated"),
            "worst_config_average_reward": final_evaluation.get("worst_config_average_reward"),
            "worst_10pct_average_reward": final_evaluation.get("worst_10pct_average_reward"),
            "per_config_metrics": final_evaluation.get("per_config_metrics"),
            "learning_rate": resolved_training_config.learning_rate,
            "gamma": resolved_training_config.gamma,
            "state_dim": state_dim,
            "num_actions": num_actions,
            "checkpoint_path": str(best_checkpoint_path),
            "plot_path": str(plot_path),
            "log_path": str(log_path),
            "evaluation_log_path": str(evaluation_log_path),
            "game_config_path": str(game_config_output_path),
            "training_config_path": str(training_config_output_path),
            "rule_signature": compute_rule_signature(resolved_game_config),
            "training_signature": compute_training_signature(resolved_training_config),
            "resume_checkpoint_path": str(resume_from) if resume_from else None,
            "resume_mode": resume_mode if resume_from else None,
            "resume_episodes_mode": resume_episodes_mode if resume_from else None,
            "resume_checkpoint_episode": (resume_metadata or {}).get("resume_checkpoint_episode"),
            "resume_restored_optimizer": (resume_metadata or {}).get("resume_restored_optimizer"),
            "resume_restored_replay_buffer": (resume_metadata or {}).get("resume_restored_replay_buffer"),
            "resume_replay_buffer_size": (resume_metadata or {}).get("resume_replay_buffer_size"),
            "resume_training_steps": (resume_metadata or {}).get("resume_training_steps"),
            "resume_checkpoint_rule_signature": (resume_metadata or {}).get("resume_checkpoint_rule_signature"),
            "resume_target_rule_signature": (resume_metadata or {}).get("resume_target_rule_signature"),
            "resume_legacy_checkpoint": (resume_metadata or {}).get("resume_legacy_checkpoint"),
            "final_epsilon": epsilon_by_episode(
                final_episode - 1,
                final_episode - 1,
                epsilon_start=resolved_training_config.epsilon_start,
                epsilon_min=resolved_training_config.epsilon_min,
                epsilon_decay_end=resolved_training_config.epsilon_decay_end,
            ),
            "mean_training_reward": (sum(training_rewards) / len(training_rewards)) if training_rewards else None,
            "mean_training_loss": (sum(training_losses) / len(training_losses)) if training_losses else None,
            "best_intermediate_evaluation_reward": best_average_reward,
            "run_notes": resolved_training_config.run_notes,
            "run_path": str(run_path) if run_path is not None else None,
        },
        str(report_dir / "dqn_metrics.json"),
    )

    return best_agent, training_rewards, final_evaluation, best_checkpoint_path, plot_path, log_path, evaluation_log_path


def parse_args():
    """Parse CLI options for dashboard-driven and manual runs."""
    parser = argparse.ArgumentParser(description="Train the advanced Double DQN Scrum Game agent.")
    parser.add_argument("--run-dir", default=None, help="Optional timestamped run directory path.")
    parser.add_argument("--game-config", default=None, help="Optional path to a game_config.json file.")
    parser.add_argument("--training-config", default=None, help="Optional path to a training_config.json file.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes.")
    parser.add_argument("--evaluation-episodes", type=int, default=None, help="Episodes per periodic checkpoint evaluation.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Checkpoint save interval.")
    parser.add_argument("--evaluation-interval", type=int, default=None, help="Periodic evaluation interval.")
    parser.add_argument("--seed", type=int, default=None, help="Training seed.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate.")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=None, help="Episodes over which epsilon decays to minimum.")
    parser.add_argument("--rule-randomization", action="store_true", help="Train on a sampled compatible rule config instead of one fixed config.")
    parser.add_argument("--rule-randomization-frequency", type=int, default=None, help="Episodes between sampled rule configs.")
    parser.add_argument("--rule-randomization-eval-configs", type=int, default=None, help="Held-out sampled configs per evaluation.")
    parser.add_argument("--rule-randomization-bounds-json", default=None, help="JSON object overriding rule-randomization bounds.")
    parser.add_argument("--auto-continue", action="store_true", help="Request automatic continuation cycles after this run completes.")
    parser.add_argument("--auto-continue-cycles", type=int, default=None, help="Maximum automatic continuation cycles; 0 means unlimited when enabled.")
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path for resume or fine-tune training.")
    parser.add_argument(
        "--resume-mode",
        choices=["strict", "fine-tune"],
        default="strict",
        help="Use 'strict' for same-rule resume or 'fine-tune' to allow rule-signature mismatch when shapes match.",
    )
    parser.add_argument(
        "--resume-episodes-mode",
        choices=["incremental", "absolute"],
        default="incremental",
        help=(
            "For resumed runs: 'incremental' treats --episodes as additional episodes, "
            "'absolute' treats --episodes as the final absolute episode target."
        ),
    )
    parser.add_argument("--agent-type", choices=["agile", "waterfall"], default="agile", help="Which agent type to train.")
    parser.add_argument("--notes", default="", help="Optional run notes saved with the artifacts.")
    return parser.parse_args()


def main():
    """Train the advanced Double DQN agent with the requested production hyperparameters."""
    args = parse_args()
    rule_randomization_bounds = json.loads(args.rule_randomization_bounds_json) if args.rule_randomization_bounds_json else None
    _, training_rewards, final_evaluation, checkpoint_path, plot_path, log_path, evaluation_log_path = train_dqn_agent(
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        checkpoint_interval=args.checkpoint_interval,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
        epsilon_decay_end=args.epsilon_decay_end,
        rule_randomization_enabled=True if args.rule_randomization else None,
        rule_randomization_frequency=args.rule_randomization_frequency,
        rule_randomization_eval_configs=args.rule_randomization_eval_configs,
        rule_randomization_bounds=rule_randomization_bounds,
        auto_continue_enabled=True if args.auto_continue else None,
        auto_continue_cycles=args.auto_continue_cycles,
        run_dir=args.run_dir,
        run_notes=args.notes,
        game_config_path=args.game_config,
        training_config_path=args.training_config,
        resume_from=args.resume_from,
        resume_mode=args.resume_mode,
        resume_episodes_mode=args.resume_episodes_mode,
        agentType=args.agent_type,
    )

    print(f"Training episodes completed: {len(training_rewards)}")
    print(f"Evaluation episodes completed: {len(final_evaluation['rewards'])}")
    print(f"Average reward per episode during evaluation: {final_evaluation['average_reward']:.2f}")
    print(f"Average ending money during evaluation: {final_evaluation['average_ending_money']:.2f}")
    print(f"Bankruptcy rate during evaluation: {final_evaluation['bankruptcy_rate']:.3f}")
    print(f"Saved best model checkpoint to: {checkpoint_path}")
    print(f"Saved training curve to: {plot_path}")
    print(f"Saved live training log to: {log_path}")
    print(f"Saved periodic evaluation log to: {evaluation_log_path}")


if __name__ == "__main__":
    main()

