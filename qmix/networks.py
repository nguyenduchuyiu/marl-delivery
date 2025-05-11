#networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random




SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

class AgentNetwork(nn.Module):
    def __init__(self, observation_shape, action_dim):
        super(AgentNetwork, self).__init__()
        # observation_shape is (C, H, W)
        self.conv1 = nn.Conv2d(observation_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        flat_size = 32 * observation_shape[1] * observation_shape[2]  # 32 * H * W

        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, obs):
        # obs: (N, C, H, W) or (C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)  # (1, C, H, W)
        x = F.relu(self.bn1(self.conv1(obs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(start_dim=1)  # (N, 32*H*W)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
    

class HyperNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, hidden_dim):
        super().__init__()
        # input_shape: (C, H, W)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        flat_size = 16 * input_shape[1] * input_shape[2]
        self.fc1 = nn.Linear(flat_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        # state: (B, C, H, W) or (C, H, W)
        if state.dim() == 3:
            state = state.unsqueeze(0)  # (1, C, H, W)
        x = state  # (B, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 32, H, W)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 16, H, W)
        x = x.flatten(start_dim=1)  # (B, 16*H*W)
        x = F.relu(self.fc1(x))     # (B, hidden_dim)
        weights = self.fc2(x)       # (B, output_dim)
        return weights


class MixingNetwork(nn.Module):
    def __init__(self, state_dim, num_agents, mixing_dim):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.mixing_dim = mixing_dim

        # state_dim: (C, H, W)
        self.hyper_w1 = HyperNetwork(state_dim, num_agents * mixing_dim, 64)
        self.hyper_b1 = HyperNetwork(state_dim, mixing_dim, 64)
        self.hyper_w2 = HyperNetwork(state_dim, mixing_dim, 64)
        self.hyper_b2 = HyperNetwork(state_dim, 1, 64)

        # Add batch normalization for state input
        self.bn = nn.BatchNorm2d(state_dim[0])

    def forward(self, agent_qs, states):
        # agent_qs: (B, num_agents) or (num_agents,)
        # states: (B, C, H, W) or (C, H, W)

        # Ensure batch dimension for agent_qs
        if agent_qs.dim() == 1:
            agent_qs = agent_qs.unsqueeze(0)  # (1, num_agents)

        batch_size = agent_qs.size(0)

        # Ensure batch dimension for states
        if states.dim() == 3:
            states = states.unsqueeze(0)  # (1, C, H, W)

        # Apply batch normalization to state input
        states = self.bn(states)

        # agent_qs: (B, num_agents) -> (B, 1, num_agents)
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)

        # First layer weights and biases
        w1 = torch.abs(self.hyper_w1(states))  # (B, num_agents * mixing_dim)
        w1 = w1.view(batch_size, self.num_agents, self.mixing_dim)  # (B, num_agents, mixing_dim)
        b1 = self.hyper_b1(states)  # (B, mixing_dim)
        b1 = b1.view(batch_size, 1, self.mixing_dim)  # (B, 1, mixing_dim)

        # Compute first layer output
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (B, 1, mixing_dim)

        # Second layer weights and biases
        w2 = torch.abs(self.hyper_w2(states))  # (B, mixing_dim)
        w2 = w2.view(batch_size, self.mixing_dim, 1)  # (B, mixing_dim, 1)
        b2 = self.hyper_b2(states)  # (B, 1)
        b2 = b2.view(batch_size, 1, 1)  # (B, 1, 1)

        # Compute final output
        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        q_tot = q_tot.squeeze(-1).squeeze(-1)  # (B,)

        return q_tot


class ReplayBuffer:
    def __init__(self, capacity, num_agents, obs_shape, state_shape, device="cpu"):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_shape = obs_shape  
        self.state_shape = state_shape 
        self.device = device

        # Determine actual shapes for numpy arrays
        _obs_s = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        _state_s = (state_shape,) if isinstance(state_shape, int) else state_shape

        self.states = np.zeros((capacity, *_state_s), dtype=np.float32)
        self.observations = np.zeros((capacity, num_agents, *_obs_s), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.total_reward = np.zeros((capacity, 1), dtype=np.float32) # Global reward
        self.next_states = np.zeros((capacity, *_state_s), dtype=np.float32)
        self.next_observations = np.zeros((capacity, num_agents, *_obs_s), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_) # Global done flag

        self.ptr = 0
        self.size = 0

    def add(self, state, next_state, obs, next_obs, actions, total_reward, done):
        """
        Adds a transition to the buffer.
        - state: np.array with shape self.state_shape
        - obs: np.array with shape (self.num_agents, *self.obs_shape)
        - actions: np.array with shape (self.num_agents,)
        - total_reward: float or np.array shape (1,) (global reward)
        - next_state: np.array with shape self.state_shape
        - next_obs: np.array with shape (self.num_agents, *self.obs_shape)
        - done: bool or np.array shape (1,) (global done flag)
        """
        self.states[self.ptr] = state
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.total_reward[self.ptr] = total_reward
        self.next_states[self.ptr] = next_state
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done # Expects global done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.
        """
        if self.size == 0:
            print("Warning: Buffer is empty. Returning empty tensors.")
            _obs_s_runtime = (self.obs_shape,) if isinstance(self.obs_shape, int) else self.obs_shape
            _state_s_runtime = (self.state_shape,) if isinstance(self.state_shape, int) else self.state_shape

            empty_states = torch.empty((0, *_state_s_runtime), dtype=torch.float32, device=self.device)
            empty_obs = torch.empty((0, self.num_agents, *_obs_s_runtime), dtype=torch.float32, device=self.device)
            empty_actions = torch.empty((0, self.num_agents), dtype=torch.long, device=self.device)
            empty_total_rewards = torch.empty((0, 1), dtype=torch.float32, device=self.device)
            empty_next_states = torch.empty((0, *_state_s_runtime), dtype=torch.float32, device=self.device)
            empty_next_obs = torch.empty((0, self.num_agents, *_obs_s_runtime), dtype=torch.float32, device=self.device)
            empty_dones = torch.empty((0, 1), dtype=torch.float32, device=self.device) # Global done
            return (empty_states, empty_obs, empty_actions, empty_total_rewards,
                    empty_next_states, empty_next_obs, empty_dones)

        if self.size < batch_size:
            print(f"Warning: Buffer size ({self.size}) is less than batch size ({batch_size}). Sampling all available data.")
            indices = np.arange(self.size)
            current_batch_size = self.size
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)

        batch_states = torch.tensor(self.states[indices], dtype=torch.float32).to(self.device)
        batch_obs = torch.tensor(self.observations[indices], dtype=torch.float32).to(self.device)
        batch_actions = torch.tensor(self.actions[indices], dtype=torch.long).to(self.device)
        batch_total_rewards = torch.tensor(self.total_reward[indices], dtype=torch.float32).to(self.device)
        batch_next_states = torch.tensor(self.next_states[indices], dtype=torch.float32).to(self.device)
        batch_next_obs = torch.tensor(self.next_observations[indices], dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device) # bool to float

        return (batch_states, batch_next_states, batch_obs, batch_next_obs, 
                batch_actions, batch_total_rewards, batch_dones)
        
    def can_sample(self, batch_size):
        return self.size >= batch_size

    def __len__(self):
        return self.size

def convert_state(state, persistent_packages, current_robot_idx=None):
    """
    Convert state to a 2D multi-channel tensor.
    - If current_robot_idx is not None: 8 channels (robot-specific obs)
    - If current_robot_idx is None: 6 channels (global state)
    Returns:
        np.ndarray of shape (n_channels, n_rows, n_cols)
    """
    grid = np.array(state["map"])
    n_rows, n_cols = grid.shape
    n_channels = 8 if current_robot_idx is not None else 6
    tensor = np.zeros((n_channels, n_rows, n_cols), dtype=np.float32)

    # 0. Map channel
    tensor[0] = grid

    # 1. Urgency channel - uses persistent_packages (for 'waiting' packages at their start location)
    current_time_step = state["time_step"][0] if isinstance(state["time_step"], np.ndarray) else state["time_step"]
    for pkg_id, pkg_data in persistent_packages.items():
        if pkg_data['status'] == 'waiting':
            sr, sc = pkg_data['start_pos'] # 0-indexed
            st = pkg_data['start_time']
            dl = pkg_data['deadline']
            urgency = max(0, min(1, (current_time_step - st) / (dl - st))) if dl > st else 0
            if 0 <= sr < n_rows and 0 <= sc < n_cols: # Boundary check
                tensor[1, sr, sc] = urgency

    # 2. Start position channel - uses persistent_packages (presence of 'waiting' packages)
    for pkg_id, pkg_data in persistent_packages.items():
        if pkg_data['status'] == 'waiting':
            sr, sc = pkg_data['start_pos'] # 0-indexed
            if 0 <= sr < n_rows and 0 <= sc < n_cols: # Boundary check
                tensor[2, sr, sc] = 1.0 # Mark presence

    # 3. Target position channel - uses persistent_packages (presence of 'waiting' or 'in_transit' package targets)
    for pkg_id, pkg_data in persistent_packages.items():
        if pkg_data['status'] in ['waiting', 'in_transit']:
            tr, tc = pkg_data['target_pos'] # 0-indexed
            if 0 <= tr < n_rows and 0 <= tc < n_cols: # Boundary check
                tensor[3, tr, tc] = 1.0 # Mark presence

    # 4. Robot carrying channel - uses state["robots"] (presence of any robot carrying a package)
    # state["robots"] provides (pos_x+1, pos_y+1, carrying_package_id) - 1-indexed
    for rob_idx, rob in enumerate(state["robots"]):
        rr, rc, carrying_pkg_id = rob
        rr_idx, rc_idx = int(rr)-1, int(rc)-1 # Convert to 0-indexed for tensor
        if carrying_pkg_id != 0: # If carrying a package
            if 0 <= rr_idx < n_rows and 0 <= rc_idx < n_cols: # Boundary check
                tensor[4, rr_idx, rc_idx] = 1.0 # Mark presence

    # 5. Robots' positions channel
    if current_robot_idx is not None: # Robot-specific: Other robots' positions
        for i, rob in enumerate(state["robots"]):
            if i != current_robot_idx:
                rr, rc, _ = rob
                rr_idx, rc_idx = int(rr)-1, int(rc)-1 # 0-indexed
                if 0 <= rr_idx < n_rows and 0 <= rc_idx < n_cols: # Boundary check
                    tensor[5, rr_idx, rc_idx] = 1.0
    else: # Global state: all robots' positions
        for rob in state["robots"]:
            rr, rc, _ = rob
            rr_idx, rc_idx = int(rr)-1, int(rc)-1 # 0-indexed
            if 0 <= rr_idx < n_rows and 0 <= rc_idx < n_cols: # Boundary check
                tensor[5, rr_idx, rc_idx] = 1.0

    # --- Robot-specific channels (if current_robot_idx is not None) ---
    if current_robot_idx is not None:
        # 6. Current robot's position channel
        rob = state["robots"][current_robot_idx]
        rr, rc, _ = rob
        rr_idx, rc_idx = int(rr)-1, int(rc)-1 # 0-indexed
        if 0 <= rr_idx < n_rows and 0 <= rc_idx < n_cols: # Boundary check
            tensor[6, rr_idx, rc_idx] = 1.0

        # 7. Current Robot's Carried Package Target Position channel
        current_robot_data = state["robots"][current_robot_idx]
        carried_pkg_id_by_current_robot = current_robot_data[2] # 1-indexed ID, 0 if not carrying

        if carried_pkg_id_by_current_robot != 0:
            # Ensure the package ID from state['robots'] is valid and exists in persistent_packages
            if carried_pkg_id_by_current_robot in persistent_packages:
                pkg_data_carried = persistent_packages[carried_pkg_id_by_current_robot]
                if pkg_data_carried['status'] == 'in_transit': # Make sure it's indeed in transit
                    tr_carried, tc_carried = pkg_data_carried['target_pos'] # 0-indexed
                    if 0 <= tr_carried < n_rows and 0 <= tc_carried < n_cols: # Boundary check
                        tensor[7, tr_carried, tc_carried] = 1.0
            # else:
                # This case might indicate an inconsistency if a robot is carrying an ID
                # not in persistent_packages. Should be handled by _update_persistent_packages.
                # print(f"Warning: Robot {current_robot_idx} carrying pkg {carried_pkg_id_by_current_robot} not in persistent_packages.")


    return tensor

