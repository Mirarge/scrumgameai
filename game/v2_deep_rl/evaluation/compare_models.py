from game.v2_deep_rl.play.match_runner import (
    ModelController,
    start_parallel_match,
    run_full_auto_match,
    build_standings_dataframe,
)

from game.v2_deep_rl.control_center.backend.services.play_service import _resolve_game_config

from game.v2_deep_rl.rl.checkpoint_utils import load_agent_for_inference

game_config = _resolve_game_config("default_game_config")[0]

agent_a, _, _ = load_agent_for_inference(
    checkpoint_path="./game/v2_deep_rl/artifacts/runs/waterfall_training/checkpoints/best_scrum_model.pth",
    agentType="waterfall"
)

agent_b, _, _ = load_agent_for_inference(
    checkpoint_path="./game/v2_deep_rl/artifacts/runs/waterfall_training/checkpoints/best_scrum_model.pth",
    agentType="waterfall"
)

baseline_wins = 0
model_wins = 0
for i in range(800):
    match_state = start_parallel_match(
        game_config,
        [
            ModelController(agent=agent_a, profile_name="random", display_name="Baseline"),
            ModelController(agent=agent_b, profile_name="expert", display_name="Trained Model"),
        ],
        base_seed=i,
    )

    match_state = run_full_auto_match(match_state)
    df = build_standings_dataframe(match_state)
    if df["Controller"][0] == "Trained Model":
        model_wins += 1
    else:
        baseline_wins += 1

print(f"Baseline wins: {baseline_wins}")
print(f"Model wins: {model_wins}")