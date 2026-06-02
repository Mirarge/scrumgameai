from waterfall_agent import WaterfallAgent
from sarsa_agent import SarsaAgent
from scrum_game_env import ScrumGameEnv

env = ScrumGameEnv()
waterfall_agent = WaterfallAgent()
sarsa_agent = SarsaAgent()

episodes = 100000
waterfall_wins = []
sarsa_wins = []
waterfall_total_rewards = []
sarsa_total_rewards = []
waterfall_to_sarsa_ratio = []

for ep in range(episodes):
    state = env.reset(seed = ep)
    done = False
    waterfall_episode_reward = 0
    sarsa_episode_reward = 0
    while not done:
        waterfall_action = waterfall_agent.choose_action(state)
        sarsa_action = sarsa_agent.choose_action(state, epsilon=5)
        waterfall_result = env.step(waterfall_action)
        sarsa_result = env.step(sarsa_action)
        waterfall_episode_reward += waterfall_result[1]
        sarsa_episode_reward += sarsa_result[1]
        state = waterfall_result[0]
        if waterfall_result[2] and sarsa_result[2]:
            done = True

    if waterfall_episode_reward > sarsa_episode_reward:
        waterfall_wins.append(ep)
        if (sarsa_episode_reward == 0):
            sarsa_episode_reward = 1
        waterfall_to_sarsa_ratio.append(abs(waterfall_episode_reward) / abs(sarsa_episode_reward))
    elif sarsa_episode_reward > waterfall_episode_reward:
        sarsa_wins.append(ep)
    waterfall_total_rewards.append(waterfall_episode_reward)
    sarsa_total_rewards.append(sarsa_episode_reward)

print("episodes:", episodes)
print("waterfall wins:", len(waterfall_wins))
print("waterfall to sarsa reward ratio (avg):", sum(waterfall_to_sarsa_ratio) / len(waterfall_to_sarsa_ratio))
print("waterfall to sarsa reward ratio (min):", min(waterfall_to_sarsa_ratio))
print("waterfall to sarsa reward ratio (max):", max(waterfall_to_sarsa_ratio))
print("sarsa wins:", len(sarsa_wins))
print("avg reward:", sum(waterfall_total_rewards) / episodes)
print("min reward:", min(waterfall_total_rewards))
print("max reward:", max(waterfall_total_rewards))
print("avg reward:", sum(sarsa_total_rewards) / episodes)
print("min reward:", min(sarsa_total_rewards))
print("max reward:", max(sarsa_total_rewards))