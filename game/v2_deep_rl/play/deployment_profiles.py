import math
import random


PROFILE_CONFIGS = {
    "beginner": {
        "epsilon": 0.15,
        "temperature": 1.6,
    },
    "balanced": {
        "epsilon": 0.05,
        "temperature": 1.0,
    },
    "expert": {
        "epsilon": 0.0,
        "temperature": 0.0,
    },
    "random": {
        "epsilon": 2.0,
        "temperature": 0.0,
    }
}


def _available_actions(agent, valid_actions=None):
    if valid_actions is None:
        return list(range(agent.num_actions))
    actions = [int(action) for action in valid_actions]
    return actions or list(range(agent.num_actions))


def choose_profile_action(agent, state_vector, profile_name, valid_actions=None):
    """Choose an action according to one of the deployment profiles."""
    profile_key = profile_name.lower()
    if profile_key not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile: {profile_name}")

    profile = PROFILE_CONFIGS[profile_key]
    epsilon = profile["epsilon"]
    temperature = profile["temperature"]
    if agent is not None:
        return agent.choose_action(state_vector, epsilon, valid_actions=valid_actions)
    actions = _available_actions(agent, valid_actions)

    if random.random() < epsilon:
        return random.choice(actions)

    q_values = agent.predict_q_values(state_vector)

    if temperature <= 0:
        return max(actions, key=lambda action: q_values[action])

    scaled = [q_values[action] / temperature for action in actions]
    max_scaled = max(scaled)
    weights = [math.exp(value - max_scaled) for value in scaled]
    return random.choices(actions, weights=weights, k=1)[0]
