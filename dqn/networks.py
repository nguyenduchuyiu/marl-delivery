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


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device="cpu"):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device

        _obs_s = (obs_shape,) if isinstance(obs_shape, int) else obs_shape

        self.observations = np.zeros((capacity, *_obs_s), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *_obs_s), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """
        Adds a transition to the buffer.
        - obs: np.array with shape self.obs_shape
        - action: int
        - reward: float
        - next_obs: np.array with shape self.obs_shape
        - done: bool
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            print("Warning: Buffer is empty. Returning empty tensors.")
            _obs_s_runtime = (self.obs_shape,) if isinstance(self.obs_shape, int) else self.obs_shape
            empty_obs = torch.empty((0, *_obs_s_runtime), dtype=torch.float32, device=self.device)
            empty_actions = torch.empty((0,), dtype=torch.long, device=self.device)
            empty_rewards = torch.empty((0,), dtype=torch.float32, device=self.device)
            empty_next_obs = torch.empty((0, *_obs_s_runtime), dtype=torch.float32, device=self.device)
            empty_dones = torch.empty((0,), dtype=torch.float32, device=self.device)
            return (empty_obs, empty_actions, empty_rewards, empty_next_obs, empty_dones)

        if self.size < batch_size:
            print(f"Warning: Buffer size ({self.size}) is less than batch size ({batch_size}). Sampling all available data.")
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)

        batch_obs = torch.tensor(self.observations[indices], dtype=torch.float32).to(self.device)
        batch_actions = torch.tensor(self.actions[indices], dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device)
        batch_next_obs = torch.tensor(self.next_observations[indices], dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device)

        return (batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones)

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def __len__(self):
        return self.size

def convert_state(state, persistent_packages, current_robot_idx):
    """
    Convert state to a 2D multi-channel tensor for a specific robot.
    - 6 channels for robot-specific observation:
        0. Map
        1. Urgency of 'waiting' packages (if robot is not carrying)
        2. Start positions of 'waiting' packages (if robot is not carrying)
        3. Other robots' positions
        4. Current robot's position
        5. Current robot's carried package target (if robot is carrying)

    Args:
        state (dict): Raw state from the environment.
                      Expected keys: "map", "robots", "time_step".
                      state["robots"] is a list of tuples: (pos_x+1, pos_y+1, carrying_package_id)
        persistent_packages (dict): Dictionary tracking all active packages.
                                    Positions are 0-indexed.
        current_robot_idx (int): Index of the current robot for which to generate the observation.

    Returns:
        np.ndarray of shape (6, n_rows, n_cols)
    """
    grid = np.array(state["map"])
    n_rows, n_cols = grid.shape
    n_channels = 6
    tensor = np.zeros((n_channels, n_rows, n_cols), dtype=np.float32)

    # --- Channel 0: Map ---
    tensor[0] = grid

    current_time_step = state["time_step"]
    if isinstance(current_time_step, np.ndarray): # Handle case where time_step might be an array
        current_time_step = current_time_step[0]

    # Get current robot's data and determine if it's carrying a package
    # Ensure current_robot_idx is valid
    if current_robot_idx < 0 or current_robot_idx >= len(state["robots"]):
        # This case should ideally be handled by the caller or indicate an error
        # print(f"Warning: Invalid current_robot_idx {current_robot_idx}")
        return tensor # Return empty tensor or handle error appropriately

    current_robot_data = state["robots"][current_robot_idx]
    carried_pkg_id_by_current_robot = current_robot_data[2] # 1-indexed ID, 0 if not carrying

    # --- Channel 1: Urgency of 'waiting' packages (if robot is not carrying) ---
    # --- Channel 2: Start positions of 'waiting' packages (if robot is not carrying) ---
    if carried_pkg_id_by_current_robot == 0: # Robot is NOT carrying a package
        for pkg_id, pkg_data in persistent_packages.items():
            if pkg_data['status'] == 'waiting':
                sr, sc = pkg_data['start_pos']  # 0-indexed
                st = pkg_data['start_time']
                dl = pkg_data['deadline']

                # Check if package is active (start_time has passed)
                if current_time_step >= st:
                    # Channel 1: Urgency
                    urgency = 0
                    if dl > st: # Avoid division by zero or negative duration
                        # Normalize urgency: 0 (just appeared) to 1 (deadline reached)
                        # Cap at 1 if current_time_step exceeds deadline
                        urgency = min(1.0, max(0.0, (current_time_step - st) / (dl - st)))
                    elif dl == st: # Deadline is the start time
                         urgency = 1.0 if current_time_step >= st else 0.0
                    # else: dl < st, invalid, urgency remains 0

                    if 0 <= sr < n_rows and 0 <= sc < n_cols: # Boundary check
                        tensor[1, sr, sc] = max(tensor[1, sr, sc], urgency) # Use max if multiple pkgs at same spot

                    # Channel 2: Start position
                    if 0 <= sr < n_rows and 0 <= sc < n_cols: # Boundary check
                        tensor[2, sr, sc] = 1.0 # Mark presence
    # If robot is carrying, channels 1 and 2 remain all zeros.

    # --- Channel 3: Other robots' positions ---
    for i, rob_data in enumerate(state["robots"]):
        if i == current_robot_idx:
            continue # Skip the current robot
        rr, rc, _ = rob_data # Positions are 1-indexed from env
        rr_idx, rc_idx = int(rr) - 1, int(rc) - 1 # Convert to 0-indexed
        if 0 <= rr_idx < n_rows and 0 <= rc_idx < n_cols: # Boundary check
            tensor[3, rr_idx, rc_idx] = 1.0

    # --- Channel 4: Current robot's position ---
    # current_robot_data was fetched earlier
    crr, crc, _ = current_robot_data # Positions are 1-indexed
    crr_idx, crc_idx = int(crr) - 1, int(crc) - 1 # Convert to 0-indexed
    if 0 <= crr_idx < n_rows and 0 <= crc_idx < n_cols: # Boundary check
        tensor[4, crr_idx, crc_idx] = 1.0

    # --- Channel 5: Current robot's carried package target (if robot is carrying) ---
    if carried_pkg_id_by_current_robot != 0:
        # Ensure the package ID from state['robots'] is valid and exists in persistent_packages
        if carried_pkg_id_by_current_robot in persistent_packages:
            pkg_data_carried = persistent_packages[carried_pkg_id_by_current_robot]
            # Double check status, though if robot carries it, it should be 'in_transit'
            # or just became 'in_transit' in the persistent_packages update logic.
            # For this observation, we primarily care about its target.
            tr_carried, tc_carried = pkg_data_carried['target_pos'] # 0-indexed
            if 0 <= tr_carried < n_rows and 0 <= tc_carried < n_cols: # Boundary check
                tensor[5, tr_carried, tc_carried] = 1.0
        # else:
            # This case might indicate an inconsistency.
            # print(f"Warning: Robot {current_robot_idx} carrying pkg {carried_pkg_id_by_current_robot} not in persistent_packages.")
    # If robot is not carrying, channel 5 remains all zeros.

    return tensor

# Define shaping reward/penalty constants
SHAPING_SUCCESSFUL_PICKUP_BONUS = 1
SHAPING_SUCCESSFUL_DELIVERY_BONUS = 10
SHAPING_LATE_DELIVERY_PENALTY = -1  # Additional penalty for being late, on top of env's
SHAPING_WASTED_PICKUP_PENALTY = -1 # Tried to pick from an empty spot or already carrying
SHAPING_WASTED_DROP_PENALTY = -1   # Tried to drop when not carrying

def reward_shaping(global_r, prev_state, current_state, actions_taken, num_agents):
    """
    Shapes the global reward 'global_r' to produce individual rewards.
    Requires prev_state to infer agent-specific events.

    Args:
        global_r (float): The global reward from the environment for the current step (s -> s').
        prev_state (dict): The state 's' before actions_taken.
        current_state (dict): The state 's'' after actions_taken.
        actions_taken (list): List of actions [action_agent_0, action_agent_1, ...]
                              taken by agents that led from prev_state to current_state.
                              Each action is (move_idx, package_op_idx).

    Returns:
        list: A list of shaped rewards, one for each agent.
    """
    
    individual_rewards = [0.0] * num_agents

    # Initialize individual rewards. Option A: Each gets the global reward.
    for i in range(num_agents):
        individual_rewards[i] = global_r

    # For easier access to package details by ID from the previous state
    prev_packages_dict = {pkg[0]: pkg for pkg in prev_state['packages']}
    current_time = current_state['time_step']

    for i in range(num_agents):
        agent_action = actions_taken[i]
        package_op = agent_action[1]  # 0: None, 1: Pick, 2: Drop

        prev_robot_info = prev_state['robots'][i]
        current_robot_info = current_state['robots'][i]

        prev_carrying_id = prev_robot_info[2]
        current_carrying_id = current_robot_info[2]

        # 1. Shaping for PICKUP attempts
        if package_op == 1:  # Agent attempted to PICKUP
            if prev_carrying_id == 0 and current_carrying_id != 0:
                # Successfully picked up a package
                individual_rewards[i] += SHAPING_SUCCESSFUL_PICKUP_BONUS
            elif prev_carrying_id != 0 : # Tried to pick up while already carrying
                individual_rewards[i] += SHAPING_WASTED_PICKUP_PENALTY
            elif prev_carrying_id == 0 and current_carrying_id == 0:
                # Attempted pickup but failed to pick up anything.
                # Check if there was a package at the robot's previous location.
                robot_prev_pos = (prev_robot_info[0], prev_robot_info[1])
                package_was_available = False
                for pkg_id, sr, sc, tr, tc, st, dl in prev_state['packages']:
                    if (sr, sc) == robot_prev_pos:
                        # A package was at the location. Maybe another agent took it, or it was a valid attempt.
                        # No penalty here, or a smaller one for contention if desired.
                        package_was_available = True
                        break
                if not package_was_available:
                    # Tried to pick up from a location with no package
                    individual_rewards[i] += SHAPING_WASTED_PICKUP_PENALTY


        # 2. Shaping for DROP attempts
        elif package_op == 2:  # Agent attempted to DROP
            if prev_carrying_id != 0 and current_carrying_id == 0:
                # Successfully delivered/dropped a package
                individual_rewards[i] += SHAPING_SUCCESSFUL_DELIVERY_BONUS
                
                # Check for timeliness if it was a delivery (package removed from list)
                delivered_pkg_id = prev_carrying_id
                # Check if this package ID is no longer in current_state['packages']
                # (indicating it was a final delivery to target)
                is_final_delivery = True
                for pkg_data in current_state['packages']:
                    if pkg_data[0] == delivered_pkg_id:
                        is_final_delivery = False # Package still exists, maybe dropped not at target
                        break
                
                if is_final_delivery and delivered_pkg_id in prev_packages_dict:
                    pkg_deadline = prev_packages_dict[delivered_pkg_id][6]
                    if current_time > pkg_deadline:
                        individual_rewards[i] += SHAPING_LATE_DELIVERY_PENALTY
            elif prev_carrying_id == 0:
                # Tried to drop when not carrying anything
                individual_rewards[i] += SHAPING_WASTED_DROP_PENALTY
                
    return individual_rewards