from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from config.config_manager import (
    load_game_config,
    save_game_config,
    compute_rule_signature,
    GameConfig,
)
from game_rules.rule_randomization import sample_game_config, merged_rule_randomization_bounds
from training.train_dqn import train_dqn_agent
from play.match_runner import start_parallel_match, run_full_auto_match
from rl.checkpoint_utils import save_checkpoint, build_agent_for_config
from rl.checkpoint_utils import build_agent_for_config as _build_agent


def play_head_to_head(game_config: GameConfig, agent_a, agent_b, games: int, base_seed: int = 1000) -> dict[str, Any]:
    wins_a = 0
    wins_b = 0
    draws = 0
    for i in range(games):
        seed = base_seed + i
        # create two model controllers by creating match state and running a full auto match
        from play.match_runner import ModelController

        controllers = [ModelController(agent=agent_a, display_name="Agent A"), ModelController(agent=agent_b, display_name="Agent B")]
        match = start_parallel_match(game_config=game_config, controllers=controllers, base_seed=seed)
        match = run_full_auto_match(match)
        # determine winner by standings
        standings = match["seats"]
        # higher ending money indicates winner
        ending = [seat["state"]["current_money"] for seat in standings]
        if ending[0] > ending[1]:
            wins_a += 1
        elif ending[1] > ending[0]:
            wins_b += 1
        else:
            draws += 1

    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws, "games": games}


def main():
    parser = argparse.ArgumentParser(description="Search for promising randomized rulesets by head-to-head matches.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--game-config", default=None)
    parser.add_argument("--initial-games", type=int, default=10)
    parser.add_argument("--max-rulesets", type=int, default=50)
    parser.add_argument("--promising-target", type=float, default=0.7)
    parser.add_argument("--leniency", type=float, default=0.05)
    parser.add_argument("--retrain-episodes", type=int, default=500)
    parser.add_argument("--max-promising", type=int, default=5)
    parser.add_argument("--final-choices", type=int, default=1)
    parser.add_argument("--final-train-episodes", type=int, default=1000)
    parser.add_argument("--bounds-json", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir

    base_game_config = load_game_config(args.game_config)
    tried_path = artifacts_dir / "tried_rulesets.json"
    tried = set()
    if tried_path.exists():
        try:
            tried = set(json.loads(tried_path.read_text(encoding="utf-8")))
        except Exception:
            tried = set()

    bounds = json.loads(args.bounds_json) if args.bounds_json else None
    rng = random.Random(42)
    promising: list[dict[str, Any]] = []

    for index in range(args.max_rulesets):
        sampled = sample_game_config(base_game_config, rng, bounds=bounds, config_name=f"random_{index+1}")
        sig = compute_rule_signature(sampled)
        if sig in tried:
            continue
        tried.add(sig)
        # quick test: build fresh agents for this config (untrained) and play initial games
        agent_a, _ = build_agent_for_config(sampled, agentType="agile")
        agent_b, _ = build_agent_for_config(sampled, agentType="waterfall")

        initial = play_head_to_head(sampled, agent_a, agent_b, games=args.initial_games, base_seed=2000 + index * 100)
        winrate_a = initial["wins_a"] / max(1, initial["games"])
        score = abs(winrate_a - args.promising_target)

        candidate = {
            "signature": sig,
            "game_config": sampled.to_dict(),
            "initial": initial,
            "score": score,
        }

        # retrain both agents for retrain_episodes
        # create unique run folder per candidate
        candidate_run = artifacts_dir / f"candidate_{index+1}"
        candidate_run.mkdir(parents=True, exist_ok=True)
        # train Agile
        train_dqn_agent(
            num_episodes=args.retrain_episodes,
            run_dir=str(candidate_run),
            game_config=sampled,
            seed=42,
            agentType="agile",
        )
        # load best checkpoints from candidate_run/checkpoints
        ag_checkpoint = candidate_run / "checkpoints" / "best_scrum_model.pth"
        # train Waterfall
        train_dqn_agent(
            num_episodes=args.retrain_episodes,
            run_dir=str(candidate_run),
            game_config=sampled,
            seed=42,
            agentType="waterfall",
        )

        # after retraining, run extended matches (X*10 - X more -> total X*10)
        total_games = max(1, args.initial_games * 10)
        more_games = total_games - args.initial_games
        # build agents from latest best checkpoints if available, else from scratch
        # For simplicity, reuse untrained agents if checkpoints missing
        agent_a2, _ = build_agent_for_config(sampled, agentType="agile")
        agent_b2, _ = build_agent_for_config(sampled, agentType="waterfall")
        try:
            from rl.checkpoint_utils import load_agent_from_checkpoint
            if ag_checkpoint.exists():
                agent_a2, _, _ = load_agent_from_checkpoint(str(ag_checkpoint), game_config=sampled, strict_signature=False)
        except Exception:
            pass

        extended = play_head_to_head(sampled, agent_a2, agent_b2, games=total_games, base_seed=3000 + index * 100)
        winrate_a_final = extended["wins_a"] / max(1, extended["games"])
        candidate.update({"extended": extended, "final_winrate_a": winrate_a_final})

        # if still close to target, save game config
        if abs(winrate_a_final - args.promising_target) <= args.leniency:
            cfg_path = artifacts_dir / f"promising_{len(promising)+1}_config.json"
            save_game_config(sampled, cfg_path)
            candidate["saved_config_path"] = str(cfg_path)
            promising.append(candidate)

        # persist tried
        tried_list = list(tried)
        tried_path.write_text(json.dumps(tried_list, indent=2), encoding="utf-8")

        # stop early if enough promising found
        if args.max_promising and len(promising) >= args.max_promising:
            break

    # after testing loop, pick top D promising for final training
    promising_sorted = sorted(promising, key=lambda x: x.get("score", 9999))
    final_candidates = promising_sorted[: args.final_choices]
    final_results = []
    for idx, cand in enumerate(final_candidates):
        cfg = GameConfig.from_dict(cand["game_config"]) if isinstance(cand.get("game_config"), dict) else cand["game_config"]
        run_final = artifacts_dir / f"final_{idx+1}"
        run_final.mkdir(parents=True, exist_ok=True)
        train_dqn_agent(num_episodes=args.final_train_episodes, run_dir=str(run_final), game_config=cfg)
        final_results.append({"signature": cand["signature"], "final_run": str(run_final)})

    out = {
        "promising": promising_sorted,
        "final": final_results,
    }
    out_path = artifacts_dir / "ruleset_search_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Ruleset search complete. Results written to {out_path}")


if __name__ == "__main__":
    main()
