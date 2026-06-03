from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import altair as alt
import pandas as pd
import streamlit as st

from config.config_manager import compute_rule_signature, load_game_config
from game_runtime.scrum_game_env import ScrumGameEnv
from play.match_runner import (
    HeuristicController,
    HumanController,
    ModelController,
    RandomController,
    all_seats_done,
    build_match_log_dataframe,
    build_standings_dataframe,
    play_round,
    run_full_auto_match,
    start_parallel_match,
    valid_actions_for_state,
)
from rl.checkpoint_utils import load_agent_from_checkpoint
from game.v2_deep_rl.rl.dqn_agile_agent import encode_state
from training.train_dqn import create_timestamped_run_directory


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
RUNS_DIR = ARTIFACTS_DIR / "runs"
ROBUSTNESS_DIR = ARTIFACTS_DIR / "robustness"
CONFIG_LIBRARY_DIR = BASE_DIR / "configs" / "custom"
TRAINING_CONFIG_LIBRARY_DIR = BASE_DIR / "configs" / "training"
LOG_PATH = REPORTS_DIR / "logs.csv"
EVALUATION_LOG_PATH = REPORTS_DIR / "evaluation_history.csv"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_scrum_model.pth"
PROCESS_STATE_PATH = REPORTS_DIR / "training_process.json"
TRAINING_STDOUT_PATH = REPORTS_DIR / "training_stdout.log"


def ensure_runtime_dirs():
    """Ensure the runtime directories used by the dashboard exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_CONFIG_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(file_path):
    """Load JSON from disk and return None if the file is missing or malformed."""
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def save_json_file(file_path, payload):
    """Persist JSON payload to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def remove_process_state():
    """Remove persisted training process metadata."""
    if PROCESS_STATE_PATH.exists():
        PROCESS_STATE_PATH.unlink()


def is_pid_running(pid):
    """Return True when a process id exists on the host."""
    if pid is None or pid <= 0:
        return False

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_training_process_state():
    """Return persisted training state only if the process is still alive."""
    process_state = load_json_file(PROCESS_STATE_PATH)
    if not process_state:
        return None

    pid = int(process_state.get("pid", 0))
    if is_pid_running(pid):
        return process_state

    remove_process_state()
    return None


def choose_python_command():
    """Choose a Python executable that exists on the current host."""
    if os.name == "nt":
        return sys.executable
    return shutil.which("python3") or sys.executable


def tail_log(file_path, max_lines=20):
    """Return the tail of a log file for launch errors."""
    if not file_path.exists():
        return ""

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            return "\n".join(handle.read().splitlines()[-max_lines:])
    except OSError:
        return ""


def launch_training_process(
    episode_count=500000,
    run_notes="",
    game_config_path="",
    training_config_path="",
    resume_from="",
    resume_mode="strict",
    learning_rate=None,
    gamma=None,
    seed=None,
    evaluation_episodes=None,
):
    """Start the DDQN trainer as an asynchronous background process."""
    ensure_runtime_dirs()
    active_process = get_training_process_state()
    if active_process:
        return False, f"Training already running (PID: {active_process['pid']}).", None

    run_dir = create_timestamped_run_directory()
    for name in ("checkpoints", "plots", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "reports" / "training_stdout.log"

    with stdout_path.open("ab") as stdout_handle:
        popen_kwargs = {
            "cwd": BASE_DIR,
            "stdout": stdout_handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True

        command = [
            choose_python_command(),
            "-m",
            "training.train_dqn",
            "--run-dir",
            str(run_dir),
            "--episodes",
            str(int(episode_count)),
            "--notes",
            run_notes or "",
        ]
        if game_config_path:
            command.extend(["--game-config", str(game_config_path)])
        if training_config_path:
            command.extend(["--training-config", str(training_config_path)])
        if resume_from:
            command.extend(["--resume-from", str(resume_from), "--resume-mode", str(resume_mode)])
        if learning_rate is not None:
            command.extend(["--learning-rate", str(learning_rate)])
        if gamma is not None:
            command.extend(["--gamma", str(gamma)])
        if seed is not None:
            command.extend(["--seed", str(int(seed))])
        if evaluation_episodes is not None:
            command.extend(["--evaluation-episodes", str(int(evaluation_episodes))])

        process = subprocess.Popen(command, **popen_kwargs)

    time.sleep(0.5)
    if process.poll() is not None:
        launch_error = tail_log(stdout_path)
        if launch_error:
            return False, f"Training process exited immediately.\n\n{launch_error}", None
        return False, "Training process exited immediately.", None

    process_state = {
        "pid": process.pid,
        "started_at_epoch": time.time(),
        "command": command,
        "stdout_log": str(stdout_path),
        "run_name": run_dir.name,
        "job_type": "training",
    }
    save_json_file(PROCESS_STATE_PATH, process_state)
    return True, f"Training launched in background (PID: {process.pid}).", run_dir.name


def launch_robustness_process(episode_count=500000):
    """Start the DDQN robustness evaluation as an asynchronous background process."""
    ensure_runtime_dirs()
    active_process = get_training_process_state()
    if active_process:
        return False, f"A background job is already running (PID: {active_process['pid']})."

    stdout_path = ROBUSTNESS_DIR / f"robustness_{time.strftime('%Y-%m-%d_%H%M')}.log"
    with stdout_path.open("ab") as stdout_handle:
        popen_kwargs = {
            "cwd": BASE_DIR,
            "stdout": stdout_handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            [
                choose_python_command(),
                "-m",
                "evaluation.evaluate_ddqn_robustness",
                "--episodes",
                str(int(episode_count)),
            ],
            **popen_kwargs,
        )

    time.sleep(0.5)
    if process.poll() is not None:
        launch_error = tail_log(stdout_path)
        if launch_error:
            return False, f"Robustness process exited immediately.\n\n{launch_error}"
        return False, "Robustness process exited immediately."

    process_state = {
        "pid": process.pid,
        "started_at_epoch": time.time(),
        "command": [choose_python_command(), "-m", "evaluation.evaluate_ddqn_robustness"],
        "stdout_log": str(stdout_path),
        "run_name": stdout_path.stem,
        "job_type": "robustness",
    }
    save_json_file(PROCESS_STATE_PATH, process_state)
    return True, f"Robustness evaluation launched in background (PID: {process.pid})."


def stop_training_process():
    """Terminate the tracked background training process."""
    process_state = get_training_process_state()
    if not process_state:
        remove_process_state()
        return False, "No active training process was found."

    pid = int(process_state["pid"])

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            return False, f"Could not stop training process {pid}: {error}"

        if result.returncode not in (0, 128):
            message = (result.stderr or result.stdout or "").strip()
            return False, f"Could not stop training process {pid}: {message or 'taskkill failed'}"
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            remove_process_state()
            return False, f"Training process {pid} was already gone."
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as error:
                return False, f"Could not stop training process {pid}: {error}"

    deadline = time.time() + 10
    while time.time() < deadline:
        if not is_pid_running(pid):
            remove_process_state()
            return True, f"Training process {pid} stopped."
        time.sleep(0.2)

    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError as error:
                return False, f"Training process {pid} did not stop cleanly: {error}"

    remove_process_state()
    return True, f"Training process {pid} was force-killed."


def format_elapsed_time(seconds_elapsed):
    """Render elapsed seconds as a compact human-readable duration."""
    total_seconds = int(max(seconds_elapsed, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def load_training_log():
    """Load the live training CSV if it exists."""
    if not LOG_PATH.exists():
        return None

    training_log = pd.read_csv(LOG_PATH)
    fallback_columns = {
        "average_loan_duration": 0.0,
        "bankruptcy_count": 0.0,
        "average_ending_money": 0.0,
        "invalid_action_count": 0.0,
    }
    for column_name, fallback_value in fallback_columns.items():
        if column_name not in training_log.columns:
            training_log[column_name] = fallback_value

    return training_log


def load_evaluation_log():
    """Load the periodic evaluation history if it exists."""
    if not EVALUATION_LOG_PATH.exists():
        return None
    return pd.read_csv(EVALUATION_LOG_PATH)


def build_empty_training_log(game_config=None):
    """Create an empty training-log dataframe so the dashboard layout stays visible."""
    base_columns = [
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
    action_columns = [
        f"action_{action_id}_count"
        for action_id in range(ScrumGameEnv(game_config=game_config).num_actions)
    ]
    return pd.DataFrame(columns=base_columns + action_columns)


def build_empty_evaluation_log():
    """Create an empty evaluation dataframe so evaluation charts stay visible."""
    return pd.DataFrame(
        columns=[
            "episode",
            "average_reward",
            "bankruptcy_rate",
            "average_ending_money",
            "average_loan_duration",
            "invalid_action_rate",
        ]
    )


def load_source_game_config(selected_source):
    """Load the game config associated with the selected run source."""
    config_path = selected_source.get("game_config_path")
    if config_path and Path(config_path).exists():
        return load_game_config(config_path)
    return load_game_config()


def list_saved_game_configs():
    """List bundled and custom-discovered game config files."""
    configs = {
        "Default Bundled Config": BASE_DIR / "configs" / "default_game_config.json",
    }
    if CONFIG_LIBRARY_DIR.exists():
        for config_path in sorted(CONFIG_LIBRARY_DIR.glob("*.json")):
            configs[config_path.name] = config_path
    return configs


def list_saved_training_configs():
    """List bundled and custom-discovered training config files."""
    configs = {
        "Default Bundled Training Config": BASE_DIR / "configs" / "default_training_config.json",
    }
    if TRAINING_CONFIG_LIBRARY_DIR.exists():
        for config_path in sorted(TRAINING_CONFIG_LIBRARY_DIR.glob("*.json")):
            configs[config_path.name] = config_path
    return configs


def list_run_sources():
    """List the legacy flat artifacts and all timestamped run folders."""
    sources = [
        {
            "label": "Current Artifacts",
            "source_id": "current",
            "checkpoints_dir": CHECKPOINT_DIR,
            "checkpoint_path": CHECKPOINT_DIR / "best_scrum_model.pth",
            "log_path": REPORTS_DIR / "logs.csv",
            "evaluation_log_path": REPORTS_DIR / "evaluation_history.csv",
            "metrics_path": REPORTS_DIR / "dqn_metrics.json",
            "notes_path": REPORTS_DIR / "run_metadata.json",
            "game_config_path": ARTIFACTS_DIR / "game_config.json",
            "training_config_path": ARTIFACTS_DIR / "training_config.json",
        }
    ]

    run_dirs = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    for run_dir in run_dirs:
        sources.append(
            {
                "label": run_dir.name,
                "source_id": run_dir.name,
                "checkpoints_dir": run_dir / "checkpoints",
                "checkpoint_path": run_dir / "checkpoints" / "best_scrum_model.pth",
                "log_path": run_dir / "reports" / "logs.csv",
                "evaluation_log_path": run_dir / "reports" / "evaluation_history.csv",
                "metrics_path": run_dir / "reports" / "dqn_metrics.json",
                "notes_path": run_dir / "run_metadata.json",
                "game_config_path": run_dir / "game_config.json",
                "training_config_path": run_dir / "training_config.json",
            }
        )

    return sources


def list_checkpoints():
    """List available checkpoints for sidebar selection."""
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
    best_first = []
    for checkpoint in checkpoints:
        if checkpoint.name == "best_scrum_model.pth":
            best_first.insert(0, checkpoint)
        else:
            best_first.append(checkpoint)
    return best_first


def list_checkpoints_for_source(source):
    """List checkpoints for one run source with best-model preference."""
    checkpoints_dir = source["checkpoints_dir"]
    if not checkpoints_dir.exists():
        return []

    checkpoints = sorted(checkpoints_dir.glob("*.pth"))
    best_first = []
    for checkpoint in checkpoints:
        if checkpoint.name == "best_scrum_model.pth":
            best_first.insert(0, checkpoint)
        else:
            best_first.append(checkpoint)
    return best_first


def load_dqn_policy(checkpoint_path, game_config=None):
    """Load a selected DDQN checkpoint into a greedy policy."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path
    if not checkpoint_path.exists():
        return None, None, None, None

    try:
        agent, env, metadata = load_agent_from_checkpoint(
            checkpoint_path,
            game_config=game_config,
            strict_signature=game_config is not None,
        )
        return agent, env, metadata, None
    except Exception as error:
        return None, None, None, str(error)


def action_label(action_id):
    """Convert an action id to a compact readable label."""
    if action_id == 0:
        return "Continue"
    return f"Switch P{action_id}"


def action_short_label(action_id):
    """Convert an action id to a short heatmap label."""
    if action_id == 0:
        return "C"
    return f"P{action_id}"


def build_strategy_map(agent, game_config, current_money=25000):
    """Create a heatmap-ready board dataframe from the current DDQN policy."""
    env = ScrumGameEnv(game_config=game_config)
    cells = []

    for product_id in range(1, env.products_count + 1):
        for sprint_id in range(1, env.sprints_per_product + 1):
            state = env.build_reference_state(
                product_id=product_id,
                sprint_id=sprint_id,
                current_money=current_money,
            )
            state_vector = encode_state(state, env)
            q_values = agent.predict_q_values(state_vector)

            best_action = max(range(env.num_actions), key=lambda action: q_values[action])
            ranked_q_values = sorted(q_values, reverse=True)
            second_best = ranked_q_values[1] if len(ranked_q_values) > 1 else ranked_q_values[0]
            confidence = q_values[best_action] - second_best

            cells.append(
                {
                    "product": env.product_names[product_id - 1],
                    "product_id": product_id,
                    "sprint": f"Sprint {sprint_id}",
                    "sprint_id": sprint_id,
                    "preferred_action_id": best_action,
                    "preferred_action": action_label(best_action),
                    "label": action_short_label(best_action),
                    "confidence": confidence,
                    "q_continue": q_values[0],
                    "best_q_value": q_values[best_action],
                    "win_probability": state["win_probability"],
                    "expected_value": state["expected_value"],
                    "remaining_turns": state["remaining_turns"],
                    "incident_active": state["incident_active"],
                    "current_incident_id": state["current_incident_id"],
                }
            )

    return pd.DataFrame(cells)


def render_strategy_heatmap(strategy_df):
    """Render the 7 x 4 board heatmap with preferred actions."""
    product_sort = strategy_df["product"].drop_duplicates().tolist()
    sprint_sort = strategy_df["sprint"].drop_duplicates().tolist()
    heatmap = (
        alt.Chart(strategy_df)
        .mark_rect()
        .encode(
            x=alt.X("product:N", sort=product_sort, title="Current Product"),
            y=alt.Y("sprint:N", sort=sprint_sort, title="Current Sprint"),
            color=alt.Color("confidence:Q", scale=alt.Scale(scheme="yellowgreenblue"), title="Policy Confidence"),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("best_q_value:Q", format=".2f"),
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("win_probability:Q", format=".3f"),
                alt.Tooltip("expected_value:Q", format=".2f"),
                "current_incident_id",
            ],
        )
        .properties(height=300)
    )

    labels = (
        alt.Chart(strategy_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x=alt.X("product:N", sort=product_sort),
            y=alt.Y("sprint:N", sort=sprint_sort),
            text="label:N",
        )
    )

    st.altair_chart(heatmap + labels, use_container_width=True)


def render_switch_target_heatmap(strategy_df):
    """Render only the cells where the agent prefers to switch away."""
    switch_df = strategy_df[strategy_df["preferred_action_id"] > 0].copy()

    if switch_df.empty:
        st.info("The selected checkpoint currently prefers Continue on all reference cells.")
        return

    product_sort = switch_df["product"].drop_duplicates().tolist()
    sprint_sort = switch_df["sprint"].drop_duplicates().tolist()
    heatmap = (
        alt.Chart(switch_df)
        .mark_rect()
        .encode(
            x=alt.X("product:N", sort=product_sort, title="Current Product"),
            y=alt.Y("sprint:N", sort=sprint_sort, title="Current Sprint"),
            color=alt.Color("preferred_action_id:Q", scale=alt.Scale(scheme="teals"), title="Preferred Switch Target"),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("confidence:Q", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    labels = (
        alt.Chart(switch_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x=alt.X("product:N", sort=product_sort),
            y=alt.Y("sprint:N", sort=sprint_sort),
            text="label:N",
        )
    )

    st.altair_chart(heatmap + labels, use_container_width=True)


def render_probability_impact_plot(strategy_df):
    """Render the win-probability vs continue-value correlation scatter plot."""
    scatter = (
        alt.Chart(strategy_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("win_probability:Q", title="Win Probability"),
            y=alt.Y("q_continue:Q", title="Q-Value for Continue"),
            color=alt.Color("preferred_action:N", title="Preferred Action"),
            tooltip=[
                "product",
                "sprint",
                "preferred_action",
                alt.Tooltip("win_probability:Q", format=".3f"),
                alt.Tooltip("q_continue:Q", format=".2f"),
                alt.Tooltip("best_q_value:Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(scatter, use_container_width=True)


def render_action_frequency_chart(training_log):
    """Render a line chart showing how often each action appears in the logs."""
    action_columns = [column for column in training_log.columns if column.startswith("action_") and column.endswith("_count")]
    if not action_columns:
        st.info("No per-action counts found in the training log yet.")
        return

    renamed = {}
    for column in action_columns:
        action_id = int(column.split("_")[1])
        renamed[column] = action_label(action_id)

    action_history = training_log[["episode"] + action_columns].rename(columns=renamed)
    melted = action_history.melt(id_vars="episode", var_name="Action", value_name="Count")

    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("Count:Q", title="Action Count per 100 Episodes"),
            color=alt.Color("Action:N"),
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_invalid_action_chart(training_log):
    """Render invalid action counts over time."""
    if "invalid_action_count" not in training_log.columns:
        st.info("No invalid action counts found in the training log yet.")
        return

    chart = (
        alt.Chart(training_log)
        .mark_line(color="#ff6b6b")
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("invalid_action_count:Q", title="Invalid Actions per 100 Episodes"),
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def run_live_demo(agent, game_config, seed=42):
    """Play one greedy rollout with the selected DDQN checkpoint."""
    env = ScrumGameEnv(game_config=game_config)
    state = env.reset(seed=seed)
    state_vector = encode_state(state, env)
    done = False
    steps = []
    total_reward = 0
    turn_number = 1

    while not done:
        q_values = agent.predict_q_values(state_vector)
        action = max(range(env.num_actions), key=lambda action_id: q_values[action_id])
        next_state, reward, done, info = env.step(action)
        next_state_vector = encode_state(next_state, env)

        steps.append(
            {
                "Turn": turn_number,
                "Product": f"Product {state['current_product']}",
                "Sprint": state["current_sprint"],
                "Win Probability": round(state["win_probability"], 3),
                "Expected Value": round(state["expected_value"], 2),
                "Action": action_label(action),
                "Outcome": info["result"],
                "Reward": reward,
                "Bank": next_state["current_money"],
            }
        )

        total_reward += reward
        state = next_state
        state_vector = next_state_vector
        turn_number += 1

    return pd.DataFrame(steps), total_reward


def find_best_demo_seed(agent, game_config, search_count=20):
    """Search a small range of seeds and return the strongest demo rollout."""
    best_seed = 0
    best_reward = float("-inf")

    for seed in range(search_count):
        _, total_reward = run_live_demo(agent, game_config=game_config, seed=seed)
        if total_reward > best_reward:
            best_reward = total_reward
            best_seed = seed

    return best_seed, best_reward


def clear_match_state():
    """Drop any existing interactive match state from the Streamlit session."""
    for key in ("parallel_match_state", "parallel_match_key"):
        if key in st.session_state:
            del st.session_state[key]


def controller_from_label(label, agent):
    """Create one match controller from a dashboard label."""
    if label == "Checkpoint Expert":
        return ModelController(agent=agent, profile_name="expert", display_name="Checkpoint Expert")
    if label == "Checkpoint Balanced":
        return ModelController(agent=agent, profile_name="balanced", display_name="Checkpoint Balanced")
    if label == "Checkpoint Beginner":
        return ModelController(agent=agent, profile_name="beginner", display_name="Checkpoint Beginner")
    if label == "Heuristic":
        return HeuristicController(display_name="Heuristic AI")
    if label == "Random":
        return RandomController(display_name="Random AI")
    raise ValueError(f"Unknown controller label: {label}")


def render_play_match_section(agent, game_config, checkpoint_identifier):
    """Render the parallel human-vs-AI / AI-vs-AI playable area."""
    st.subheader("Play Match")
    st.caption(
        "This match mode runs one seat per controller on the same ruleset. "
        "Each seat plays its own copy of the board, which keeps the current single-player DDQN usable."
    )

    match_seed = st.number_input("Match seed", min_value=0, value=42, step=1, key="parallel_match_seed")
    include_human = st.checkbox("Include Human Seat", value=True, key="parallel_match_human")
    opponent_labels = st.multiselect(
        "Opponents",
        ["Checkpoint Expert", "Checkpoint Balanced", "Checkpoint Beginner", "Heuristic", "Random"],
        default=["Checkpoint Expert", "Heuristic"],
        key="parallel_match_opponents",
    )

    match_key = f"{checkpoint_identifier}:{compute_rule_signature(game_config)}:{match_seed}:{include_human}:{','.join(opponent_labels)}"
    if st.session_state.get("parallel_match_key") and st.session_state["parallel_match_key"] != match_key:
        clear_match_state()

    start_col, reset_col, auto_col = st.columns(3)
    with start_col:
        if st.button("Start New Match", use_container_width=True):
            controllers = []
            if include_human:
                controllers.append(HumanController(display_name="Human"))
            controllers.extend(controller_from_label(label, agent) for label in opponent_labels)

            if not controllers:
                st.warning("Select at least one seat before starting a match.")
            else:
                match_state = start_parallel_match(
                    game_config=game_config,
                    controllers=controllers,
                    base_seed=int(match_seed),
                )
                if not include_human:
                    match_state = run_full_auto_match(match_state)
                st.session_state["parallel_match_state"] = match_state
                st.session_state["parallel_match_key"] = match_key
                st.rerun()
    with reset_col:
        if st.button("Reset Match", use_container_width=True):
            clear_match_state()
            st.rerun()
    with auto_col:
        if st.button("Auto Finish AI Match", use_container_width=True):
            match_state = st.session_state.get("parallel_match_state")
            if match_state is not None:
                human_active = any(
                    seat["controller"].controller_type == "human" and not seat["done"]
                    for seat in match_state["seats"]
                )
                if human_active:
                    st.warning("Human seats still need manual actions before the match can auto-finish.")
                else:
                    st.session_state["parallel_match_state"] = run_full_auto_match(match_state)
                    st.rerun()

    match_state = st.session_state.get("parallel_match_state")
    if not match_state:
        st.info("Start a match to compare a human seat and multiple AI controllers on the current ruleset.")
        return

    standings = build_standings_dataframe(match_state)
    st.dataframe(standings, use_container_width=True, hide_index=True)

    human_seat = next(
        (
            seat for seat in match_state["seats"]
            if seat["controller"].controller_type == "human" and not seat["done"]
        ),
        None,
    )

    if human_seat is not None:
        valid_actions = valid_actions_for_state(human_seat["env"], human_seat["state"])
        action_labels = {
            action_id: human_seat["env"].action_name(action_id)
            for action_id in valid_actions
        }
        selected_label = st.selectbox(
            "Human Action",
            list(action_labels.values()),
            key=f"parallel_human_action_{len(human_seat['steps'])}",
        )
        selected_action = next(
            action_id for action_id, label in action_labels.items() if label == selected_label
        )
        if st.button("Play Next Round", use_container_width=True):
            st.session_state["parallel_match_state"] = play_round(
                match_state,
                human_action=selected_action,
            )
            st.rerun()
    elif all_seats_done(match_state):
        st.success("Match complete.")

    match_log = build_match_log_dataframe(match_state)
    st.dataframe(match_log, use_container_width=True, hide_index=True)


def render_server_controls():
    """Render sidebar controls for starting and stopping background training."""
    st.sidebar.header("Server Controls")
    active_process = get_training_process_state()

    if active_process:
        elapsed = format_elapsed_time(time.time() - active_process.get("started_at_epoch", time.time()))
        st.sidebar.success(f"Training in Progress (PID: {active_process['pid']})")
        st.sidebar.caption(f"Elapsed: {elapsed}")
    else:
        st.sidebar.info("No background training job is currently running.")

    launch_disabled = active_process is not None
    if st.sidebar.button("🚀 Launch 500k Episode Training", use_container_width=True, disabled=launch_disabled):
        try:
            success, message = launch_training_process()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
            st.rerun()
        except Exception as error:
            st.sidebar.error(f"Could not launch training: {error}")

    if st.sidebar.button("🛑 Stop Training", use_container_width=True, disabled=active_process is None):
        success, message = stop_training_process()
        if success:
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)
        st.rerun()

    if TRAINING_STDOUT_PATH.exists():
        st.sidebar.caption(f"Trainer log: {TRAINING_STDOUT_PATH.name}")

    return active_process


def render_server_controls_v2():
    """Render sidebar controls for starting and stopping background jobs."""
    st.sidebar.header("Server Controls")
    active_process = get_training_process_state()
    saved_game_configs = list_saved_game_configs()
    saved_game_config_labels = list(saved_game_configs.keys())
    saved_training_configs = list_saved_training_configs()
    saved_training_config_labels = list(saved_training_configs.keys())
    resume_sources = {
        source["label"]: source
        for source in list_run_sources()
        if source["checkpoint_path"].exists()
    }
    resume_source_labels = ["None"] + list(resume_sources.keys())
    game_override_labels = ["None"] + saved_game_config_labels + ["Manual Path"]
    training_override_labels = ["None"] + saved_training_config_labels + ["Manual Path"]
    resume_override_labels = ["None", "Manual Path"]
    if st.session_state.get("selected_saved_game_config") not in saved_game_config_labels:
        st.session_state["selected_saved_game_config"] = saved_game_config_labels[0]
    if st.session_state.get("selected_training_config_option") not in training_override_labels:
        st.session_state["selected_training_config_option"] = "None"
    if st.session_state.get("selected_game_config_override") not in game_override_labels:
        st.session_state["selected_game_config_override"] = "None"
    if st.session_state.get("resume_source_label") not in resume_source_labels:
        st.session_state["resume_source_label"] = "None"
    if st.session_state.get("selected_resume_checkpoint_override") not in resume_override_labels:
        st.session_state["selected_resume_checkpoint_override"] = "None"

    if active_process:
        elapsed = format_elapsed_time(time.time() - active_process.get("started_at_epoch", time.time()))
        label = active_process.get("job_type", "training").title()
        st.sidebar.success(f"{label} in Progress (PID: {active_process['pid']})")
        st.sidebar.caption(f"Elapsed: {elapsed}")
    else:
        st.sidebar.info("No background training job is currently running.")

    st.sidebar.text_input("Run Notes", key="run_notes")
    st.sidebar.number_input("Episode Count", min_value=1000, step=1000, key="episode_count")
    st.sidebar.selectbox(
        "Saved Game Config",
        saved_game_config_labels,
        key="selected_saved_game_config",
    )
    st.sidebar.caption(f"Auto-read folder: {CONFIG_LIBRARY_DIR}")
    st.sidebar.selectbox(
        "Game Config Override",
        game_override_labels,
        key="selected_game_config_override",
    )
    if st.session_state["selected_game_config_override"] == "Manual Path":
        st.sidebar.text_input("Game Config Manual Path", key="game_config_path")
    st.sidebar.selectbox(
        "Training Config",
        training_override_labels,
        key="selected_training_config_option",
    )
    st.sidebar.caption(f"Auto-read folder: {TRAINING_CONFIG_LIBRARY_DIR}")
    if st.session_state["selected_training_config_option"] == "Manual Path":
        st.sidebar.text_input("Training Config Manual Path", key="training_config_path")
    st.sidebar.selectbox("Resume From Run", resume_source_labels, key="resume_source_label")
    selected_resume_checkpoint_path = ""
    if st.session_state["resume_source_label"] != "None":
        selected_resume_source = resume_sources[st.session_state["resume_source_label"]]
        resume_checkpoints = list_checkpoints_for_source(selected_resume_source)
        resume_checkpoint_labels = [checkpoint.name for checkpoint in resume_checkpoints]
        resume_checkpoint_key = f"resume_checkpoint_label::{st.session_state['resume_source_label']}"
        if st.session_state.get(resume_checkpoint_key) not in resume_checkpoint_labels:
            st.session_state[resume_checkpoint_key] = (
                resume_checkpoint_labels[0] if resume_checkpoint_labels else ""
            )
        if resume_checkpoint_labels:
            st.sidebar.selectbox(
                "Resume Checkpoint",
                resume_checkpoint_labels,
                key=resume_checkpoint_key,
            )
            selected_resume_checkpoint = next(
                checkpoint
                for checkpoint in resume_checkpoints
                if checkpoint.name == st.session_state[resume_checkpoint_key]
            )
            selected_resume_checkpoint_path = str(selected_resume_checkpoint)
            st.sidebar.caption(f"Auto-selected checkpoint: {selected_resume_checkpoint_path}")
    st.sidebar.selectbox(
        "Resume Checkpoint Override",
        resume_override_labels,
        key="selected_resume_checkpoint_override",
    )
    if st.session_state["selected_resume_checkpoint_override"] == "Manual Path":
        st.sidebar.text_input("Resume Checkpoint Manual Path", key="resume_checkpoint_path")
    st.sidebar.selectbox("Resume Mode", ["strict", "fine-tune"], key="resume_mode")
    st.sidebar.caption("Use strict for same rules. Use fine-tune only when the model shape still matches.")
    st.sidebar.number_input("Learning Rate Override", min_value=0.00001, step=0.0001, format="%.5f", key="learning_rate_override")
    st.sidebar.number_input("Gamma Override", min_value=0.1, max_value=0.9999, step=0.01, format="%.4f", key="gamma_override")
    st.sidebar.number_input("Seed Override", min_value=0, step=1, key="seed_override")
    st.sidebar.number_input("Eval Episodes Override", min_value=10, step=10, key="evaluation_episodes_override")

    launch_disabled = active_process is not None
    if st.sidebar.button("Launch 500k Episode Training", use_container_width=True, disabled=launch_disabled):
        try:
            selected_game_config_path = str(saved_game_configs[st.session_state["selected_saved_game_config"]])
            if st.session_state["selected_game_config_override"] == "Manual Path":
                selected_game_config_path = st.session_state["game_config_path"].strip() or selected_game_config_path
            elif st.session_state["selected_game_config_override"] != "None":
                selected_game_config_path = str(
                    saved_game_configs[st.session_state["selected_game_config_override"]]
                )

            selected_training_config_path = ""
            if st.session_state["selected_training_config_option"] == "Manual Path":
                selected_training_config_path = st.session_state["training_config_path"].strip()
            elif st.session_state["selected_training_config_option"] != "None":
                selected_training_config_path = str(
                    saved_training_configs[st.session_state["selected_training_config_option"]]
                )

            selected_resume_path = (
                st.session_state["resume_checkpoint_path"].strip()
                if st.session_state["selected_resume_checkpoint_override"] == "Manual Path"
                else selected_resume_checkpoint_path
            )
            success, message, run_name = launch_training_process(
                episode_count=st.session_state["episode_count"],
                run_notes=st.session_state["run_notes"],
                game_config_path=selected_game_config_path,
                training_config_path=selected_training_config_path,
                resume_from=selected_resume_path,
                resume_mode=st.session_state["resume_mode"],
                learning_rate=st.session_state["learning_rate_override"],
                gamma=st.session_state["gamma_override"],
                seed=st.session_state["seed_override"],
                evaluation_episodes=st.session_state["evaluation_episodes_override"],
            )
            if success and run_name:
                st.session_state["selected_source"] = run_name
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
            st.rerun()
        except Exception as error:
            st.sidebar.error(f"Could not launch training: {error}")

    if st.sidebar.button("Launch 5-Seed Robustness Eval", use_container_width=True, disabled=launch_disabled):
        try:
            success, message = launch_robustness_process(episode_count=st.session_state["episode_count"])
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.warning(message)
            st.rerun()
        except Exception as error:
            st.sidebar.error(f"Could not launch robustness evaluation: {error}")

    if st.sidebar.button("Stop Training", use_container_width=True, disabled=active_process is None):
        success, message = stop_training_process()
        if success:
            st.sidebar.warning(message)
        else:
            st.sidebar.error(message)
        st.rerun()

    stdout_log = active_process.get("stdout_log") if active_process else str(TRAINING_STDOUT_PATH)
    if stdout_log:
        st.sidebar.caption(f"Trainer log: {Path(stdout_log).name}")

    return active_process


ensure_runtime_dirs()
st.set_page_config(page_title="Scrum Game DDQN Dashboard", layout="wide")
st.title("Scrum Game Command Center")
st.caption("Live monitoring for the advanced 8-action Double DQN branch.")

if "run_notes" not in st.session_state:
    st.session_state["run_notes"] = ""
if "episode_count" not in st.session_state:
    st.session_state["episode_count"] = 500000
if "game_config_path" not in st.session_state:
    st.session_state["game_config_path"] = ""
if "selected_saved_game_config" not in st.session_state:
    st.session_state["selected_saved_game_config"] = "Default Bundled Config"
if "selected_game_config_override" not in st.session_state:
    st.session_state["selected_game_config_override"] = "None"
if "training_config_path" not in st.session_state:
    st.session_state["training_config_path"] = ""
if "selected_training_config_option" not in st.session_state:
    st.session_state["selected_training_config_option"] = "None"
if "resume_checkpoint_path" not in st.session_state:
    st.session_state["resume_checkpoint_path"] = ""
if "resume_source_label" not in st.session_state:
    st.session_state["resume_source_label"] = "None"
if "selected_resume_checkpoint_override" not in st.session_state:
    st.session_state["selected_resume_checkpoint_override"] = "None"
if "resume_mode" not in st.session_state:
    st.session_state["resume_mode"] = "strict"
if "learning_rate_override" not in st.session_state:
    st.session_state["learning_rate_override"] = 0.0005
if "gamma_override" not in st.session_state:
    st.session_state["gamma_override"] = 0.85
if "seed_override" not in st.session_state:
    st.session_state["seed_override"] = 42
if "evaluation_episodes_override" not in st.session_state:
    st.session_state["evaluation_episodes_override"] = 100
if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = "Current Artifacts"

active_process = render_server_controls_v2()

st.sidebar.header("Dashboard")
run_sources = list_run_sources()
source_lookup = {source["label"]: source for source in run_sources}
source_labels = [source["label"] for source in run_sources]
selected_source_label = st.sidebar.selectbox(
    "Run Folder",
    source_labels,
    index=source_labels.index(st.session_state["selected_source"]) if st.session_state["selected_source"] in source_labels else 0,
)
st.session_state["selected_source"] = selected_source_label
selected_source = source_lookup[selected_source_label]
selected_game_config = load_source_game_config(selected_source)
selected_rule_signature = compute_rule_signature(selected_game_config)

checkpoint_options = list_checkpoints() if selected_source["checkpoints_dir"] == CHECKPOINT_DIR else sorted(selected_source["checkpoints_dir"].glob("*.pth"))
checkpoint_labels = [checkpoint.name for checkpoint in checkpoint_options]
selected_checkpoint_label = st.sidebar.selectbox(
    "Checkpoint",
    checkpoint_labels if checkpoint_labels else ["best_scrum_model.pth"],
    index=0,
)
selected_checkpoint_path = (
    checkpoint_options[checkpoint_labels.index(selected_checkpoint_label)]
    if checkpoint_options
    else selected_source["checkpoint_path"]
)

auto_refresh = st.sidebar.toggle("🔄 Auto-Refresh", value=active_process is not None)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", min_value=2, max_value=30, value=5, step=1)
if st.sidebar.button("Refresh Charts", use_container_width=True):
    st.rerun()

demo_seed = st.sidebar.number_input("Demo seed", min_value=0, value=42, step=1)
scan_demo_seeds = st.sidebar.checkbox("Find best demo seed from range", value=False)
demo_search_count = st.sidebar.number_input("Demo seed scan size", min_value=5, value=20, step=5)
reference_money = st.sidebar.number_input("Heatmap reference money", min_value=0, value=25000, step=5000)

if selected_source.get("notes_path"):
    notes_data = load_json_file(selected_source["notes_path"])
    if notes_data and notes_data.get("run_notes"):
        st.info(f"Run Notes: {notes_data['run_notes']}")

download_col1, download_col2 = st.columns(2)
with download_col1:
    if selected_source["checkpoint_path"].exists():
        st.download_button(
            "Download best_scrum_model.pth",
            data=selected_source["checkpoint_path"].read_bytes(),
            file_name="best_scrum_model.pth",
            mime="application/octet-stream",
            use_container_width=True,
        )
    else:
        st.button("Download best_scrum_model.pth", disabled=True, use_container_width=True)
with download_col2:
    if selected_source["log_path"].exists():
        st.download_button(
            "Download logs.csv",
            data=selected_source["log_path"].read_bytes(),
            file_name="logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button("Download logs.csv", disabled=True, use_container_width=True)

training_log = load_training_log() if selected_source["log_path"] == LOG_PATH else (pd.read_csv(selected_source["log_path"]) if selected_source["log_path"].exists() else None)
if training_log is not None:
    fallback_columns = {
        "average_loan_duration": 0.0,
        "bankruptcy_count": 0.0,
        "average_ending_money": 0.0,
        "invalid_action_count": 0.0,
    }
    for column_name, fallback_value in fallback_columns.items():
        if column_name not in training_log.columns:
            training_log[column_name] = fallback_value

evaluation_log = load_evaluation_log() if selected_source["evaluation_log_path"] == EVALUATION_LOG_PATH else (pd.read_csv(selected_source["evaluation_log_path"]) if selected_source["evaluation_log_path"].exists() else None)
agent, policy_env, checkpoint_metadata, checkpoint_error = load_dqn_policy(
    selected_checkpoint_path,
    game_config=selected_game_config,
)

has_training_log = training_log is not None and not training_log.empty
if not has_training_log:
    st.warning(
        "No training log found yet. Launch training from the sidebar or run `py -m training.train_dqn` "
        "to start logging to `artifacts/reports/logs.csv`."
    )
    training_log = build_empty_training_log(game_config=selected_game_config)
else:
    latest = training_log.iloc[-1]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Latest Episode", int(latest["episode"]))
    metric_col2.metric("Latest Reward", f"{latest['episode_reward']:.2f}")
    metric_col3.metric("Rolling Avg Reward", f"{latest['rolling_average_reward']:.2f}")
    metric_col4.metric("Epsilon", f"{latest['epsilon']:.4f}")

    debt_col1, debt_col2, debt_col3 = st.columns(3)
    debt_col1.metric("Average Loan Duration", f"{latest['average_loan_duration']:.2f}")
    debt_col2.metric("Bankruptcy Count / 100", int(latest["bankruptcy_count"]))
    debt_col3.metric("Average Ending Money", f"{latest['average_ending_money']:.2f}")

st.caption(
    f"Active rule signature: `{selected_rule_signature}`"
)

if not has_training_log:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Latest Episode", "0")
    metric_col2.metric("Latest Reward", "0.00")
    metric_col3.metric("Rolling Avg Reward", "0.00")
    metric_col4.metric("Epsilon", "0.0000")

    debt_col1, debt_col2, debt_col3 = st.columns(3)
    debt_col1.metric("Average Loan Duration", "0.00")
    debt_col2.metric("Bankruptcy Count / 100", "0")
    debt_col3.metric("Average Ending Money", "0.00")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Training Reward")
    reward_chart = (
        alt.Chart(training_log)
        .transform_fold(["episode_reward", "rolling_average_reward"], as_=["metric", "value"])
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("value:Q", title="Reward"),
            color=alt.Color("metric:N", title="Metric"),
        )
        .properties(height=320)
    )
    st.altair_chart(reward_chart, use_container_width=True)

with chart_col2:
    st.subheader("Debt Stress")
    debt_chart = (
        alt.Chart(training_log)
        .transform_fold(
            ["average_loan_duration", "bankruptcy_count", "average_ending_money"],
            as_=["metric", "value"],
        )
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("metric:N", title="Metric"),
        )
        .properties(height=320)
    )
    st.altair_chart(debt_chart, use_container_width=True)

diag_col1, diag_col2 = st.columns(2)
with diag_col1:
    st.subheader("Recent Loss")
    loss_chart = (
        alt.Chart(training_log)
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("mean_recent_loss:Q", title="Recent Mean Loss"),
        )
        .properties(height=260)
    )
    st.altair_chart(loss_chart, use_container_width=True)

with diag_col2:
    st.subheader("Replay Buffer Size")
    buffer_chart = (
        alt.Chart(training_log)
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Episode"),
            y=alt.Y("replay_buffer_size:Q", title="Replay Buffer Size"),
        )
        .properties(height=260)
    )
    st.altair_chart(buffer_chart, use_container_width=True)

st.subheader("Per-Action Frequency")
render_action_frequency_chart(training_log)

st.subheader("Invalid Actions")
render_invalid_action_chart(training_log)

has_evaluation_log = evaluation_log is not None and not evaluation_log.empty
if not has_evaluation_log:
    evaluation_log = build_empty_evaluation_log()

eval_col1, eval_col2 = st.columns(2)
with eval_col1:
    st.subheader("Evaluation Reward And Ending Money")
    reward_chart = (
        alt.Chart(evaluation_log)
        .transform_fold(["average_reward", "average_ending_money"], as_=["metric", "value"])
        .mark_line()
        .encode(
            x=alt.X("episode:Q", title="Checkpoint Episode"),
            y=alt.Y("value:Q", title="Reward / Ending Money"),
            color=alt.Color("metric:N", title="Metric"),
        )
        .properties(height=280)
    )
    st.altair_chart(reward_chart, use_container_width=True)

with eval_col2:
    st.subheader("Evaluation Risk Metrics")
    risk_columns = [
        column
        for column in ["bankruptcy_rate", "invalid_action_rate", "average_loan_duration"]
        if column in evaluation_log.columns
    ]
    if risk_columns:
        risk_chart = (
            alt.Chart(evaluation_log)
            .transform_fold(risk_columns, as_=["metric", "value"])
            .mark_line()
            .encode(
                x=alt.X("episode:Q", title="Checkpoint Episode"),
                y=alt.Y("value:Q", title="Risk Metric"),
                color=alt.Color("metric:N", title="Metric"),
            )
            .properties(height=280)
        )
        st.altair_chart(risk_chart, use_container_width=True)

if checkpoint_error is not None:
    st.error(
        "The selected checkpoint is incompatible with the selected game config or network shape. "
        "Use a checkpoint trained for the same rule signature."
    )
elif agent is None:
    st.info("No compatible DDQN checkpoint found yet. Strategy maps and live demo will appear after training saves one.")
else:
    if checkpoint_metadata and checkpoint_metadata.get("legacy_checkpoint"):
        st.warning(
            "The selected checkpoint is a legacy file without embedded rule metadata. "
            "Compatibility is being inferred from the current environment shape only."
        )
    if checkpoint_metadata and checkpoint_metadata.get("checkpoint_rule_signature"):
        st.caption(
            f"Checkpoint rule signature: `{checkpoint_metadata['checkpoint_rule_signature']}`"
        )

    strategy_df = build_strategy_map(
        agent,
        game_config=selected_game_config,
        current_money=int(reference_money),
    )

    st.subheader("Strategy Heatmap")
    render_strategy_heatmap(strategy_df)

    st.subheader("Switch Target Heatmap")
    render_switch_target_heatmap(strategy_df)

    st.subheader("Probability vs Continue Q-Value")
    render_probability_impact_plot(strategy_df)

    st.subheader("Live Demo")
    if scan_demo_seeds:
        best_seed, best_seed_reward = find_best_demo_seed(
            agent,
            game_config=selected_game_config,
            search_count=int(demo_search_count),
        )
        st.caption(f"Best seed in 0..{int(demo_search_count) - 1}: {best_seed} (reward {best_seed_reward:.2f})")
        demo_seed = best_seed

    demo_table, total_demo_reward = run_live_demo(
        agent,
        game_config=selected_game_config,
        seed=int(demo_seed),
    )
    st.metric("Demo Total Reward", f"{total_demo_reward:.2f}")
    st.dataframe(demo_table, use_container_width=True, hide_index=True)

    render_play_match_section(
        agent,
        game_config=selected_game_config,
        checkpoint_identifier=str(selected_checkpoint_path),
    )

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
