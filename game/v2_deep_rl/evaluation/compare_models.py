from game.v2_deep_rl.play.match_runner import (
    ModelController,
    start_parallel_match,
    run_full_auto_match,
    build_standings_dataframe,
)

from game.v2_deep_rl.control_center.backend.services.play_service import _resolve_game_config

from game.v2_deep_rl.rl.checkpoint_utils import load_agent_for_inference

game_config = _resolve_game_config("default_game_config")[0]

waterfall, _, _ = load_agent_for_inference(
    checkpoint_path="./game/v2_deep_rl/artifacts/runs/waterfall_training/checkpoints/best_scrum_model.pth",
    agentType="waterfall"
)

agile, _, _ = load_agent_for_inference(
    checkpoint_path="./game/v2_deep_rl/artifacts/runs/agile_training/checkpoints/best_scrum_model.pth",
    agentType="agile"
)

agent_a_wins = 0
agent_b_wins = 0
ties = 0
bankruptcy_losses = 0

episodes = 800

agent_a_name = "Baseline"
agent_b_name = "Trained Model"

agent_a_total = 0
agent_b_total = 0

for i in range(episodes):
    match_state = start_parallel_match(
        game_config,
        [
            ModelController(agent=waterfall, profile_name="random", display_name=agent_a_name),
            ModelController(agent=waterfall, profile_name="expert", display_name=agent_b_name),
        ],
        base_seed=i,
    )

    match_state = run_full_auto_match(match_state)
    df = build_standings_dataframe(match_state)
    agent_a_reward = df.loc[df["Controller"] == agent_a_name, "Total Reward"].values[0]
    agent_b_reward = df.loc[df["Controller"] == agent_b_name, "Total Reward"].values[0]

    bankruptcy_losses += 1 if (len((df.loc[df["Terminal"] == "bankruptcy", "Terminal"]).values) > 0) else 0

    agent_a_total += agent_a_reward
    agent_b_total += agent_b_reward

    if agent_a_reward > agent_b_reward:
        agent_a_wins += 1
    elif agent_b_reward > agent_a_reward:
        agent_b_wins += 1
    else:
        ties += 1

print(f"{agent_a_name} wins: {agent_a_wins}")
print(f"{agent_a_name} average: {agent_a_total/episodes}")
print(f"{agent_a_name} total: {agent_a_total}")

print(f"{agent_b_name} wins: {agent_b_wins}")
print(f"{agent_b_name} average: {agent_b_total/episodes}")
print(f"{agent_b_name} total: {agent_b_total}")

print(f"Ties: {ties}")
print(f"Bankruptcy losses: {bankruptcy_losses}")