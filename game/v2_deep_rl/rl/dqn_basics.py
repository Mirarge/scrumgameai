from collections import deque
import random

from torch import nn

class ReplayBuffer:
    """Fixed-size replay memory for DDQN transitions."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        """Return a torch-serializable snapshot of replay memory."""
        return {
            "capacity": self.buffer.maxlen,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, state):
        """Restore replay memory from a checkpoint snapshot."""
        capacity = int(state.get("capacity") or self.buffer.maxlen or 100000)
        self.buffer = deque(state.get("buffer", []), maxlen=capacity)


class QNetwork(nn.Module):
    """MLP that maps the normalized Scrum Game state vector to action-values."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)