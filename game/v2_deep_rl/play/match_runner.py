from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from game_runtime.scrum_game_env import ScrumGameEnv
from play.deployment_profiles import choose_profile_action
from game.v2_deep_rl.rl.dqn_agile_agent import encode_state


def valid_actions_for_state(env: ScrumGameEnv, state: dict[str, Any]) -> list[int]:
    """Return the currently valid action ids for one environment state."""
    valid_actions = []
    current_product = int(state["current_product"])

    if not state["current_product_completed"]:
        valid_actions.append(0)

    for product_id in range(1, env.products_count + 1):
        if product_id == current_product:
            continue
        if state["target_is_completed"][product_id - 1]:
            continue
        valid_actions.append(product_id)

    return valid_actions or [0]


@dataclass
class Controller:
    controller_type: str
    display_name: str

    def choose_action(self, state, env) -> int:
        raise NotImplementedError


@dataclass
class HumanController(Controller):
    controller_type: str = "human"
    display_name: str = "Human"

    def choose_action(self, state, env) -> int:
        raise RuntimeError("HumanController actions must be provided by the UI.")


@dataclass
class RandomController(Controller):
    controller_type: str = "random"
    display_name: str = "Random AI"

    def choose_action(self, state, env) -> int:
        valid_actions = valid_actions_for_state(env, state)
        return random.choice(valid_actions)


@dataclass
class HeuristicController(Controller):
    controller_type: str = "heuristic"
    display_name: str = "Heuristic AI"

    def choose_action(self, state, env) -> int:
        valid_actions = valid_actions_for_state(env, state)
        if len(valid_actions) == 1:
            return valid_actions[0]

        current_score = float(state["expected_value"]) - float(env.cost_continue)
        best_action = 0
        best_score = current_score

        for action in valid_actions:
            if action == 0:
                continue
            candidate_score = float(state["target_expected_values"][action - 1]) - float(env.cost_switch_mid)
            if state["target_is_completed"][state["current_product"] - 1]:
                candidate_score += float(env.cost_switch_mid - env.cost_switch_after)
            if candidate_score > best_score:
                best_score = candidate_score
                best_action = action

        return best_action


@dataclass
class ModelController(Controller):
    agent: Any = None
    profile_name: str = "expert"
    controller_type: str = "model"
    display_name: str = "Checkpoint AI"

    def choose_action(self, state, env) -> int:
        state_vector = encode_state(state, env)
        return choose_profile_action(
            self.agent,
            state_vector,
            profile_name=self.profile_name,
            valid_actions=valid_actions_for_state(env, state),
        )


def create_match_seat(controller: Controller, game_config, seed: int) -> dict[str, Any]:
    """Create one independent seat for the parallel match runner."""
    env = ScrumGameEnv(game_config=game_config)
    initial_state = env.reset(seed=seed)
    return {
        "controller": controller,
        "seed": seed,
        "env": env,
        "state": initial_state,
        "done": False,
        "steps": [],
        "total_reward": 0.0,
        "terminal_reason": "",
    }


def start_parallel_match(game_config, controllers: list[Controller], base_seed: int = 42) -> dict[str, Any]:
    """Create one config-consistent parallel match state."""
    seats = [
        create_match_seat(controller, game_config=game_config, seed=base_seed + index)
        for index, controller in enumerate(controllers)
    ]
    return {
        "game_config": game_config,
        "base_seed": base_seed,
        "round_number": 1,
        "seats": seats,
    }


def _record_step(seat, action, reward, done, info):
    seat["steps"].append(
        {
            "Round": len(seat["steps"]) + 1,
            "Controller": seat["controller"].display_name,
            "Action": info["action_name"],
            "Outcome": info["result"],
            "Reward": reward,
            "Bank": info["ending_money"],
            "Terminal": info.get("terminal_reason", ""),
        }
    )
    seat["total_reward"] += reward
    seat["done"] = done
    seat["terminal_reason"] = info.get("terminal_reason", "")


def play_round(match_state, human_action: int | None = None) -> dict[str, Any]:
    """Advance every active seat by one turn. The human seat consumes the supplied action."""
    for seat in match_state["seats"]:
        if seat["done"]:
            continue

        controller = seat["controller"]
        env = seat["env"]
        state = seat["state"]

        if controller.controller_type == "human":
            if human_action is None:
                continue
            action = human_action
        else:
            action = controller.choose_action(state, env)

        next_state, reward, done, info = env.step(action)
        seat["state"] = next_state
        _record_step(seat, action, reward, done, info)

    match_state["round_number"] += 1
    return match_state


def all_seats_done(match_state) -> bool:
    """Return whether every seat has finished its run."""
    return all(seat["done"] for seat in match_state["seats"])


def run_full_auto_match(match_state) -> dict[str, Any]:
    """Run the match until every non-human seat is complete."""
    while not all_seats_done(match_state):
        human_seats = [
            seat for seat in match_state["seats"]
            if seat["controller"].controller_type == "human" and not seat["done"]
        ]
        if human_seats:
            break
        play_round(match_state)
    return match_state


def build_standings_dataframe(match_state) -> pd.DataFrame:
    """Create a scoreboard for the current match state."""
    import pandas as pd
    rows = []
    for seat in match_state["seats"]:
        rows.append(
            {
                "Controller": seat["controller"].display_name,
                "Type": seat["controller"].controller_type,
                "Total Reward": round(seat["total_reward"], 2),
                "Ending Money": seat["state"]["current_money"],
                "Turns Played": len(seat["steps"]),
                "Done": seat["done"],
                "Terminal": seat["terminal_reason"],
            }
        )
    standings = pd.DataFrame(rows)
    if standings.empty:
        return standings
    return standings.sort_values(
        by=["Ending Money", "Total Reward"],
        ascending=False,
    ).reset_index(drop=True)


def build_match_log_dataframe(match_state) -> pd.DataFrame:
    """Create a long-form turn log for all seats."""
    import pandas as pd
    rows = []
    for seat in match_state["seats"]:
        rows.extend(seat["steps"])
    if not rows:
        return pd.DataFrame(
            columns=["Round", "Controller", "Action", "Outcome", "Reward", "Bank", "Terminal"]
        )
    return pd.DataFrame(rows)
