import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from MAPPO.helper import convert_observation, generate_vector_features
from MAPPO.networks import ActorNetwork as AgentNetwork

NUM_MOVE_ACTIONS = 5  # S, L, R, U, D
NUM_PKG_OPS    = 3  # None, Pickup, Drop
JOINT_ACTION_DIM = NUM_MOVE_ACTIONS * NUM_PKG_OPS
        
class Agents:
    def __init__(self, observation_shape, vector_obs_dim, max_time_steps, weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Joint action encoder
        self.le_move = LabelEncoder().fit(['S','L','R','U','D'])
        self.le_pkg  = LabelEncoder().fit(['0','1','2']) # Assuming '0' is None, '1' is Pickup, '2' is Drop
        self.model = AgentNetwork(observation_shape, vector_obs_dim).to(self.device) # Pass JOINT_ACTION_DIM
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
        self.max_time_steps = max_time_steps
    def _update_persistent_packages(self, current_env_state_dict):
        # This is a simplified version of the DQNTrainer's method, adapted for one env
        current_persistent_packages = self.persistent_packages
        
        if 'packages' in current_env_state_dict and current_env_state_dict['packages'] is not None:
            for pkg_tuple in current_env_state_dict['packages']:
                pkg_id = pkg_tuple[0]
                if pkg_id not in current_persistent_packages:
                    current_persistent_packages[pkg_id] = {
                        'id': pkg_id,
                        'start_pos': (pkg_tuple[1] - 1, pkg_tuple[2] - 1),
                        'target_pos': (pkg_tuple[3] - 1, pkg_tuple[4] - 1),
                        'start_time': pkg_tuple[5],
                        'deadline': pkg_tuple[6],
                        'status': 'waiting'
                    }

        current_carried_pkg_ids_set = set()
        if 'robots' in current_env_state_dict and current_env_state_dict['robots'] is not None:
            for r_data in current_env_state_dict['robots']:
                carried_id = r_data[2]
                if carried_id != 0:
                    current_carried_pkg_ids_set.add(carried_id)

        packages_to_remove = []
        for pkg_id, pkg_data in list(current_persistent_packages.items()):
            if pkg_id in current_carried_pkg_ids_set:
                current_persistent_packages[pkg_id]['status'] = 'in_transit'
            else:
                if pkg_data['status'] == 'in_transit':
                    packages_to_remove.append(pkg_id)
        
        for pkg_id_to_remove in packages_to_remove:
            if pkg_id_to_remove in current_persistent_packages:
                del current_persistent_packages[pkg_id_to_remove]
        self.persistent_packages = current_persistent_packages

    def init_agents(self, state):
        self.n_robots = len(state.get('robots', []))
        self._update_persistent_packages(state) # Initialize/update based on initial state
        self.is_init = True

    def get_actions(self, state, deterministic=True):
        assert self.is_init, "Agents not initialized. Call init_agents(state) first."
        
        # Update persistent packages based on the current state
        self._update_persistent_packages(state)
        
        actions = []
        for i in range(self.n_robots):
            # Prepare observation and vector features
            obs = convert_observation(state, self.persistent_packages, current_robot_idx=i)
            vector_obs = generate_vector_features(
                state, 
                self.persistent_packages, 
                i, 
                self.max_time_steps,
            )
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
            vector_obs_t = torch.tensor(vector_obs, dtype=torch.float32).to(self.device)
            if obs_t.dim() == 3:
                obs_t = obs_t.unsqueeze(0)
                vector_obs_t = vector_obs_t.unsqueeze(0)
            with torch.no_grad():
                logits = self.model(obs_t, vector_obs_t)  # shape [1, JOINT_ACTION_DIM]
                if deterministic:
                    joint_idx = logits.argmax(dim=1).item()
                else:
                    probs = torch.softmax(logits, dim=1)
                    joint_idx = torch.multinomial(probs, num_samples=1).item()
            
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
