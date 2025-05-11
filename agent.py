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


class Agents:
    def __init__(self, observation_shape, weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Joint action encoder
        self.le_move = LabelEncoder().fit(['S','L','R','U','D'])
        self.le_pkg  = LabelEncoder().fit(['0','1','2']) # Assuming '0' is None, '1' is Pickup, '2' is Drop
        self.model = AgentNetwork(observation_shape, JOINT_ACTION_DIM).to(self.device) # Pass JOINT_ACTION_DIM
        if weights_path is not None:
            try:
                # It's good practice to print a message before loading
                print(f"Loading model weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}")

        self.model.eval()
        self.n_robots = 0
        self.is_init = False
        self.persistent_packages = {} # Initialize persistent_packages

    def _update_persistent_packages(self, current_env_state):
        """
        Updates self.persistent_packages based on the current environment state.
        - current_env_state: The state dictionary.
        """
        # 1. Add newly appeared packages to persistent_packages if not already tracked
        if 'packages' in current_env_state and current_env_state['packages'] is not None:
            for pkg_tuple in current_env_state['packages']:
                pkg_id = pkg_tuple[0]
                if pkg_id not in self.persistent_packages:
                    self.persistent_packages[pkg_id] = {
                        'id': pkg_id,
                        'start_pos': (pkg_tuple[1] - 1, pkg_tuple[2] - 1), # Assuming 1-indexed to 0-indexed
                        'target_pos': (pkg_tuple[3] - 1, pkg_tuple[4] - 1), # Assuming 1-indexed to 0-indexed
                        'start_time': pkg_tuple[5],
                        'deadline': pkg_tuple[6],
                        'status': 'waiting'
                    }

        # 2. Get current robot carrying info
        current_carried_pkg_ids_set = set()
        if 'robots' in current_env_state and current_env_state['robots'] is not None:
            for r_idx, r_data in enumerate(current_env_state['robots']):
                # r_data format: (pos_x+1, pos_y+1, carrying_package_id)
                carried_id = r_data[2]
                if carried_id != 0: # 0 means not carrying
                    current_carried_pkg_ids_set.add(carried_id)

        packages_to_remove_definitively = []

        # 3. Update package status
        for pkg_id, pkg_data in list(self.persistent_packages.items()): # Iterate over a copy for safe deletion
            original_status_in_tracker = pkg_data['status']

            if pkg_id in current_carried_pkg_ids_set:
                # If currently being carried by any robot in current_env_state, set to 'in_transit'
                self.persistent_packages[pkg_id]['status'] = 'in_transit'
            else:
                # Package is NOT being carried in current_env_state
                if original_status_in_tracker == 'in_transit':
                    # This package WAS 'in_transit' (according to our tracker)
                    # and is now NOT carried in current_env_state.
                    # This implies it was delivered or dropped.
                    # For simplicity here, we'll assume delivered and remove.
                    # More sophisticated logic might be needed depending on env rules (e.g., dropped vs delivered)
                    packages_to_remove_definitively.append(pkg_id)
                # If original_status_in_tracker was 'waiting' and it's still not carried,
                # its status remains 'waiting'.

        # 4. Remove packages that were processed (e.g., delivered)
        for pkg_id_to_remove in packages_to_remove_definitively:
            if pkg_id_to_remove in self.persistent_packages:
                del self.persistent_packages[pkg_id_to_remove]

    def init_agents(self, state):
        self.n_robots = len(state.get('robots', []))
        self._update_persistent_packages(state) # Initialize/update based on initial state
        self.is_init = True

    def get_actions(self, state):
        assert self.is_init, "Agents not initialized. Call init_agents(state) first."
        
        # Update persistent packages based on the current state
        self._update_persistent_packages(state)
        
        actions = []
        for i in range(self.n_robots):
            if np.random.rand() < 0.1: # Epsilon-greedy exploration (0.1 epsilon)
                joint_idx = np.random.randint(0, JOINT_ACTION_DIM)
            else:
                # Pass self.persistent_packages to convert_state
                obs = convert_state(state, self.persistent_packages, current_robot_idx=i)
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
                if obs_t.dim() == 3: # Ensure batch dimension if not present
                    obs_t = obs_t.unsqueeze(0)
                
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
