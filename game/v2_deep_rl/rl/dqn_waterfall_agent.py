import math
import random

import torch
from torch import nn

from rl.dqn_basics import QNetwork, ReplayBuffer


class WaterfallAgent:
    """Double DQN agent for the advanced Scrum Game environment that follows a waterfall strategy."""

    def __init__(
        self,
        state_dim,
        num_actions,
        learning_rate=0.0005,
        gamma=0.85,
        replay_capacity=100000,
        batch_size=128,
        target_update_frequency=2000,
        device=None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = QNetwork(input_dim=state_dim, output_dim=num_actions).to(self.device)
        self.target_network = QNetwork(input_dim=state_dim, output_dim=num_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.training_steps = 0

    def training_state_dict(self):
        """Return optimizer and replay state needed for strict continuation."""
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "training_steps": self.training_steps,
        }

    def load_training_state_dict(self, state, include_replay=True):
        """Restore mutable training state from a checkpoint snapshot."""
        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        if include_replay and state.get("replay_buffer") is not None:
            self.replay_buffer.load_state_dict(state["replay_buffer"])

        if state.get("training_steps") is not None:
            self.training_steps = int(state["training_steps"])

    def choose_action(self, state_vector, epsilon, valid_actions=None):
        q_values = self.predict_q_values(state_vector)

        if 0 in valid_actions:
            return 0
        else:
            if random.random() < epsilon:
                return random.randint(1, self.num_actions - 1)
            else:
                action = max(valid_actions, key=lambda action: q_values[action])
                return action

    def choose_action_with_temperature(self, state_vector, temperature=1.0):
        """Sample an action from the current Q-values using a softmax temperature."""
        q_values = self.predict_q_values(state_vector)

        if temperature <= 0:
            return max(range(self.num_actions), key=lambda action: q_values[action])

        scaled = [value / temperature for value in q_values]
        max_scaled = max(scaled)
        weights = [math.exp(value - max_scaled) for value in scaled]
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        return random.choices(range(self.num_actions), weights=probabilities, k=1)[0]

    def predict_q_values(self, state_vector):
        """Return the raw Q-values for one normalized state vector."""
        state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor).squeeze(0).cpu().tolist()
        return q_values

    def store_transition(self, state, action, reward, next_state, done):
        """Add one transition to replay memory."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """
        Sample a batch from replay memory and update the policy network.

        This uses the Double DQN target:
        - the policy network chooses the best next action
        - the target network evaluates that chosen action
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_network(states_tensor).gather(1, actions_tensor).squeeze(1)

        with torch.no_grad():
            next_policy_actions = self.policy_network(next_states_tensor).argmax(dim=1, keepdim=True)
            next_target_values = self.target_network(next_states_tensor).gather(1, next_policy_actions).squeeze(1)
            target_q_values = rewards_tensor + self.gamma * next_target_values * (1.0 - dones_tensor)

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return float(loss.item())


def encode_state(state, env):
    """Convert the rich environment observation into a normalized float vector."""
    current_money = float(state["current_money"]) / max(env.mandatory_loan_amount * 4, 1)
    current_product = float(state["current_product"] - 1) / max(env.products_count - 1, 1)
    current_sprint = float(state["current_sprint"]) / max(env.sprints_per_product, 1)
    features_required = float(state["features_required"]) / 4.0
    sprint_value = float(state["sprint_value"]) / max(env.max_visible_sprint_value, 1)
    loan_active = float(bool(state["loan_active"]))
    interest_due = float(state["interest_due"]) / max(env.max_interest_reference, 1)
    win_probability = float(state["win_probability"])
    expected_value = float(state["expected_value"]) / max(env.max_visible_sprint_value, 1)
    remaining_turns = float(state["remaining_turns"]) / max(env.max_turns, 1)
    is_last_sprint = float(state["is_last_sprint"])
    debt_ratio = min(max(float(state["debt_ratio"]), 0.0), 2.0) / 2.0
    switch_is_free = float(state["switch_is_free"])
    incident_active = float(state["incident_active"])
    current_incident_id = float(state["current_incident_id"]) / 500.0
    current_incident_scope = float(state["current_incident_scope"])
    current_incident_delta = float(state["current_incident_delta"]) / max(env.max_visible_sprint_value, 1)
    current_refinement_delta = (
        float(state["current_refinement_delta"]) + env.max_refinement_reference
    ) / max(env.max_refinement_reference * 2, 1)
    current_product_completed = float(state["current_product_completed"])

    vector = [
        current_money,
        current_product,
        current_sprint,
        features_required,
        sprint_value,
        loan_active,
        interest_due,
        win_probability,
        expected_value,
        remaining_turns,
        is_last_sprint,
        debt_ratio,
        switch_is_free,
        incident_active,
        current_incident_id,
        current_incident_scope,
        current_incident_delta,
        current_refinement_delta,
        current_product_completed,
    ]

    for next_sprint in state["target_next_sprints"]:
        vector.append(float(next_sprint) / max(env.sprints_per_product, 1))

    for features in state["target_features_required"]:
        vector.append(float(features) / 4.0)

    for tsv in state["target_sprint_values"]:
        vector.append(float(tsv) / max(env.max_visible_sprint_value, 1))

    for twp in state["target_win_probabilities"]:
        vector.append(float(twp))

    for tev in state["target_expected_values"]:
        vector.append(float(tev) / max(env.max_visible_sprint_value, 1))

    for completed_flag in state["target_is_completed"]:
        vector.append(float(completed_flag))

    for incident_delta in state["target_incident_deltas"]:
        vector.append(float(incident_delta) / max(env.max_visible_sprint_value, 1))

    for refinement_delta in state["target_refinement_deltas"]:
        vector.append((float(refinement_delta) + env.max_refinement_reference) / max(env.max_refinement_reference * 2, 1))

    for incident_flag in state["target_incident_flags"]:
        vector.append(float(incident_flag))

    vector.extend(_encode_rule_context(env))

    return vector


def _encode_rule_context(env):
    """Expose rule/config context so one policy can adapt across randomized rules."""
    max_money_reference = max(env.mandatory_loan_amount * 2, env.starting_money, 1)
    max_cost_reference = max(env.mandatory_loan_amount, env.cost_switch_mid, env.cost_switch_after, 1)
    max_penalty_reference = max(abs(env.penalty_negative), abs(env.penalty_positive), 1)

    features = [
        float(env.starting_money) / max_money_reference,
        float(env.max_turns) / 15.0,
        float(env.cost_continue) / max_cost_reference,
        float(env.cost_switch_mid) / max_cost_reference,
        float(env.cost_switch_after) / max_cost_reference,
        float(env.mandatory_loan_amount) / max_money_reference,
        float(env.loan_interest) / max(env.mandatory_loan_amount, 1),
        float(env.penalty_negative) / max_penalty_reference,
        float(env.penalty_positive) / max_penalty_reference,
        float(env.daily_scrums_per_sprint) / 10.0,
        float(env.daily_scrum_target) / 30.0,
        float(env.incident_draw_probability),
        float(env.incident_severity_multiplier) / 3.0,
        float(env.incidents_active),
        float(env.refinements_active),
    ]

    for rule in env.dice_rules:
        features.append(float(rule.dice_count) / 4.0)
        features.append(float(rule.dice_sides) / 20.0)

    return features
