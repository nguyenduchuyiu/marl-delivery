import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sklearn.preprocessing import LabelEncoder

class SimpleActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, n_actions=40):
        super(SimpleActorCritic, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Not used directly; use act() and evaluate()
        raise NotImplementedError

    def act(self, x):
        features = self.policy_net(x)
        action_logits = self.action_head(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        action = torch.multinomial(action_probs, 1)
        return action.squeeze(-1), action_probs

    def evaluate(self, x):
        features = self.value_net(x)
        value = self.value_head(features)
        return value

def convert_state(state):
    ret_state = {}
    ret_state["robots"] = np.array(state["robots"]).astype(np.float32).flatten()
    ret_state["packages"] = np.array(state["packages"]).astype(np.float32).flatten()[:100]
    if len(ret_state["packages"]) < 100:
        ret_state["packages"] = np.concatenate((ret_state["packages"], np.zeros(100-len(ret_state["packages"]))))
    obs = np.concatenate(list(ret_state.values()))
    # Pad to 115 if needed
    if len(obs) < 115:
        obs = np.concatenate([obs, np.zeros(115 - len(obs), dtype=np.float32)])
    return obs

class Agents:

    def __init__(self, weights_path=None):
        """
            TODO:
        """
        self.agents = []
        self.n_robots = 0
        self.state = None
        self.model = None
        if weights_path is not None:
            self.model = PPO.load(weights_path)

    def init_agents(self, state):
        """
            TODO:
        """
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']

    def get_actions(self, state):
        """
            Use the neural network to select actions for each robot.
        """
        obs = convert_state(state)
        obs = obs.reshape(1, -1)
        actions, _ = self.model.predict(obs, deterministic=True)
        print("Raw actions:", actions)
        actions = actions.reshape(-1, 2)
        print("Decoded actions:", actions)
        actions = list(zip(
            le1.inverse_transform(actions[:, 0]),
            le2.inverse_transform(actions[:, 1])
        ))
        actions = [(str(a[0]), str(a[1])) for a in actions]
        return actions

# Define these globally or in your Agents class
le1 = LabelEncoder()
le1.fit(['S', 'L', 'R', 'U', 'D'])
le2 = LabelEncoder()
le2.fit(['0', '1', '2'])
