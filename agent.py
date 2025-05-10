import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# Action space dimensions
NUM_MOVE_ACTIONS = 5  # S, L, R, U, D
NUM_PKG_OPS    = 3  # None, Pickup, Drop
JOINT_ACTION_DIM = NUM_MOVE_ACTIONS * NUM_PKG_OPS

class AgentNetwork(nn.Module):
    def __init__(self, observation_shape, action_dim=JOINT_ACTION_DIM):
        super(AgentNetwork, self).__init__()
        # observation_shape is (C, H, W)
        c, h, w = observation_shape
        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        flat_size = 32 * h * w
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, obs):
        # obs: (N, C, H, W) or (C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits for joint action


def convert_state(state, current_robot_idx=None):
    """
    Convert raw state dict to multi-channel tensor.
    If current_robot_idx is None: global state (6 channels).
    Otherwise: agent-specific (7 channels).
    """
    grid = np.array(state["map"])
    n_rows, n_cols = grid.shape
    n_ch = 7 if current_robot_idx is not None else 6
    tensor = np.zeros((n_ch, n_rows, n_cols), dtype=np.float32)
    # Map
    tensor[0] = grid
    # Urgency
    t = state.get("time_step", 0)
    for pkg in state.get("packages", []):
        pkg_id, sr, sc, tr, tc, st, dl = pkg
        sr, sc, tr, tc = int(sr)-1, int(sc)-1, int(tr)-1, int(tc)-1
        urgency = 0
        if dl > st:
            urgency = max(0.0, min(1.0, (t - st) / (dl - st)))
        tensor[1, sr, sc] = urgency
    # Start / target
    for pkg in state.get("packages", []):
        pkg_id, sr, sc, tr, tc, _, _ = pkg
        sr, sc = int(sr)-1, int(sc)-1
        tr, tc = int(tr)-1, int(tc)-1
        tensor[2, sr, sc] = pkg_id
        tensor[3, tr, tc] = -pkg_id
    # Robot carrying and positions
    for rob in state.get("robots", []):
        rr, rc, carrying = rob
        rr, rc = int(rr)-1, int(rc)-1
        if carrying != 0:
            tensor[4, rr, rc] = 1
        tensor[5, rr, rc] = 1
    # Current robot
    if current_robot_idx is not None:
        rr, rc, _ = state["robots"][current_robot_idx]
        rr, rc = int(rr)-1, int(rc)-1
        tensor[6, rr, rc] = 1
    return tensor

class Agents:
    def __init__(self, observation_shape, weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Joint action encoder
        self.le_move = LabelEncoder().fit(['S','L','R','U','D'])
        self.le_pkg  = LabelEncoder().fit(['0','1','2'])
        self.model = AgentNetwork(observation_shape).to(self.device)
        if weights_path is not None:
            print(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.n_robots = 0
        self.is_init = False

    def init_agents(self, state):
        self.n_robots = len(state.get('robots', []))
        self.is_init = True

    def get_actions(self, state):
        assert self.is_init, "Agents not initialized. Call init_agents(state) first."
        actions = []
        for i in range(self.n_robots):
            if np.random.rand() < 0.1:
                joint_idx = np.random.randint(0, JOINT_ACTION_DIM)
            else:
                obs = convert_state(state, current_robot_idx=i)
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self.model(obs_t)  # shape [1, JOINT_ACTION_DIM]
                    joint_idx = logits.argmax(dim=1).item()
            # decode joint index
            move_idx   = joint_idx % NUM_MOVE_ACTIONS
            pkg_idx    = joint_idx // NUM_MOVE_ACTIONS
            move_str   = self.le_move.inverse_transform([move_idx])[0]
            pkg_str    = self.le_pkg.inverse_transform([pkg_idx])[0]
            actions.append((move_str, pkg_str))
        return actions

# Example usage:
# obs_shape = (7, env.n_rows, env.n_cols)
# agents = Agents(obs_shape, weights_path=None)
# agents.init_agents(env.reset())
# act_list = agents.get_actions(current_state)
