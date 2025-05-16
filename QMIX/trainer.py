# QMIX/trainer.py
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from QMIX.networks import RNNAgent, QMixer, ReplayBuffer
from QMIX.helper import (
    Args, convert_observation, generate_vector_features,
    convert_global_state, # compute_shaped_rewards, # QMIX typically uses global reward
    save_qmix_model, load_qmix_model
)
from env_vectorized import VectorizedEnv # Assuming this is your vectorized environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QMixLearner:
    def __init__(self, n_agents, action_dim, args: Args,
                 vec_env: VectorizedEnv, replay_buffer: ReplayBuffer,
                 le_move: LabelEncoder, le_pkg_op: LabelEncoder):
        self.args = args
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.device = device

        self.vec_env = vec_env
        self.replay_buffer = replay_buffer
        self.le_move = le_move
        self.le_pkg_op = le_pkg_op

        # --- Initialize Networks ---
        self.agent_net = RNNAgent(
            args.spatial_obs_shape, args.vector_obs_dim, args.rnn_hidden_dim
        ).to(self.device)
        self.mixer_net = QMixer(
            self.n_agents, args.global_spatial_state_shape, args.global_vector_state_dim, args.mixing_embed_dim,
            hypernet_embed=args.hypernet_embed
        ).to(self.device)

        self.target_agent_net = RNNAgent(
            args.spatial_obs_shape, args.vector_obs_dim, args.rnn_hidden_dim
        ).to(self.device)
        self.target_mixer_net = QMixer(
            self.n_agents, args.global_spatial_state_shape, args.global_vector_state_dim, args.mixing_embed_dim
        ).to(self.device)

        self.update_target_networks(hard_update=True)

        # --- Optimizer ---
        # Combine parameters from agent and mixer networks for a single optimizer
        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = optim.RMSprop(params, lr=args.lr, alpha=args.optim_alpha)
        
        self.train_step_counter = 0 # To track when to update target networks

        # State for data collection
        self.current_env_states_list = None
        self.persistent_packages_list = [{} for _ in range(self.args.num_parallel_envs)]
        self.current_agent_hidden_states_list = None

    def _update_persistent_packages_for_env(self, env_idx, current_env_state_dict):
        current_persistent_packages = self.persistent_packages_list[env_idx]
        
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
        self.persistent_packages_list[env_idx] = current_persistent_packages

    def _initialize_collection_state(self, initial_reset=True):
        """Initializes or resets the state needed for data collection."""
        if initial_reset:
            self.current_env_states_list = self.vec_env.reset() # List of env_state_dicts

        self.persistent_packages_list = [{} for _ in range(self.args.num_parallel_envs)]
        for env_idx, initial_state_dict in enumerate(self.current_env_states_list):
            self._update_persistent_packages_for_env(env_idx, initial_state_dict)

        # Fix: Ensure hidden state shape is (self.n_agents, hidden_dim)
        hidden = self.agent_net.init_hidden()
        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)  # (1, 1, hidden_dim) -> (1, hidden_dim)
        elif hidden.dim() == 2 and hidden.shape[0] == 1:
            hidden = hidden  # (1, hidden_dim)
        # Now expand to (self.n_agents, hidden_dim)
        self.current_agent_hidden_states_list = [
            hidden.expand(self.n_agents, -1).to(self.device)
            for _ in range(self.args.num_parallel_envs)
        ]

    def update_target_networks(self, hard_update=False):
        if hard_update:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())
        else: # Soft update (Exponential Moving Average)
            for target_param, param in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)
            for target_param, param in zip(self.target_mixer_net.parameters(), self.mixer_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)

    def _perform_update(self, batch):
        # Renamed from train()
        bs_eps = batch['so'].shape[0] # batch_size (number of episodes in batch)
        T = batch['so'].shape[1]      # episode_limit (max timesteps in an episode)

        # --- Move batch data to the correct device ---
        for key in batch.keys():
            if isinstance(batch[key], np.ndarray): # Convert numpy arrays to tensors
                 batch[key] = torch.from_numpy(batch[key]).to(self.device, 
                                                              dtype=torch.float32 if batch[key].dtype == np.float32 else torch.long)
            else: # Assuming it's already a tensor, just move to device
                 batch[key] = batch[key].to(self.device)


        # --- Calculate Q-values for current actions using the online networks ---
        q_agent_all_timesteps = []
        # Initial hidden state for RNN: (bs_eps * N, rnn_hidden_dim)
        h_agent = self.agent_net.init_hidden().expand(bs_eps * self.n_agents, -1).clone()

        for t in range(T): # Iterate over timesteps in the episode
            # Reshape observations for this timestep: (bs_eps * N, feature_dims)
            so_t = batch['so'][:, t].reshape(bs_eps * self.n_agents, *self.args.spatial_obs_shape)
            vo_t = batch['vo'][:, t].reshape(bs_eps * self.n_agents, self.args.vector_obs_dim)
            
            # Get Q-values and next hidden state from RNNAgent
            q_agent_t, h_agent_next = self.agent_net(so_t, vo_t, h_agent) # q_agent_t: (bs_eps*N, n_actions)
            q_agent_all_timesteps.append(q_agent_t)
            h_agent = h_agent_next # Propagate hidden state

        # Stack Q-values across timesteps: (T, bs_eps*N, n_actions)
        # Permute and reshape to: (bs_eps*T*N, n_actions)
        q_agent_current_all = torch.stack(q_agent_all_timesteps, dim=0).permute(1,0,2)
        q_agent_current_all = q_agent_current_all.reshape(bs_eps, self.n_agents, T, self.action_dim)
        q_agent_current_all = q_agent_current_all.permute(0,2,1,3).reshape(bs_eps*T*self.n_agents, self.action_dim)

        # Get Q-values for the actions actually taken (from buffer)
        actions_batch_u = batch['u'].reshape(bs_eps * T * self.n_agents, 1).long()
        chosen_action_qvals = torch.gather(q_agent_current_all, dim=1, index=actions_batch_u).squeeze(1) # (bs_eps*T*N)
        # Reshape for mixer input: (bs_eps*T, N)
        chosen_action_qvals_for_mixer = chosen_action_qvals.reshape(bs_eps * T, self.n_agents)

        # Get total Q-value from the mixer
        global_spatial_state_batch = batch['gs'].reshape(bs_eps * T, *self.args.global_spatial_state_shape)
        global_vector_state_batch = batch['gv'].reshape(bs_eps * T, self.args.global_vector_state_dim)
        q_total_current = self.mixer_net(
            chosen_action_qvals_for_mixer, global_spatial_state_batch, global_vector_state_batch
        ) # (bs_eps*T, 1)

        # --- Calculate Target Q-values using the target networks ---
        q_target_agent_all_timesteps = []
        h_target_agent = self.target_agent_net.init_hidden().expand(bs_eps * self.n_agents, -1).clone()

        for t in range(T):
            so_next_t = batch['so_next'][:, t].reshape(bs_eps * self.n_agents, *self.args.spatial_obs_shape)
            vo_next_t = batch['vo_next'][:, t].reshape(bs_eps * self.n_agents, self.args.vector_obs_dim)
            q_target_agent_t, h_target_agent_next = self.target_agent_net(so_next_t, vo_next_t, h_target_agent)
            q_target_agent_all_timesteps.append(q_target_agent_t)
            h_target_agent = h_target_agent_next
        
        q_target_agent_all = torch.stack(q_target_agent_all_timesteps, dim=0).permute(1,0,2)
        q_target_agent_all = q_target_agent_all.reshape(bs_eps, self.n_agents, T, self.action_dim)
        q_target_agent_all = q_target_agent_all.permute(0,2,1,3).reshape(bs_eps*T*self.n_agents, self.action_dim)

        # Mask unavailable actions for target Q calculation
        next_avail_actions_batch = batch['avail_u_next'].reshape(bs_eps * T * self.n_agents, self.action_dim).bool()
        q_target_agent_all[~next_avail_actions_batch] = -float('inf') 
        
        # Select best action for target Q (max_a Q_target_a)
        q_target_max_actions = q_target_agent_all.max(dim=1)[0] # (bs_eps*T*N)
        q_target_max_for_mixer = q_target_max_actions.reshape(bs_eps * T, self.n_agents) # (bs_eps*T, N)

        # Get total target Q-value from the target mixer
        next_global_spatial_state_batch = batch['gs_next'].reshape(bs_eps * T, *self.args.global_spatial_state_shape)
        next_global_vector_state_batch = batch['gv_next'].reshape(bs_eps * T, self.args.global_vector_state_dim)
        q_total_target_next = self.target_mixer_net(
            q_target_max_for_mixer, next_global_spatial_state_batch, next_global_vector_state_batch
        ) # (bs_eps*T, 1)

        # --- Calculate TD Target and Loss ---
        rewards_batch = batch['r'].reshape(bs_eps * T, 1)
        terminated_batch = batch['terminated'].reshape(bs_eps * T, 1)
        # y = r + gamma * Q_tot_target_next * (1 - terminated)
        targets_td = rewards_batch + self.args.gamma * q_total_target_next * (1 - terminated_batch)
        
        td_error = (q_total_current - targets_td.detach()) # Detach targets_td to prevent gradients from flowing into target nets
        
        # Mask out padded steps (steps beyond episode termination within the episode_limit)
        mask = (1 - batch['padded'].reshape(bs_eps*T, 1)).float()
        masked_td_error = td_error * mask
        
        # Loss: Mean Squared Error over non-padded steps
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(self.agent_net.parameters()) + list(self.mixer_net.parameters()), self.args.grad_norm_clip)
        self.optimizer.step()
        torch.cuda.empty_cache()
        self.train_step_counter += 1
        if self.train_step_counter % self.args.target_update_interval == 0:
            self.update_target_networks(hard_update=self.args.hard_target_update) # Use configured update type
        return loss.item()

    def run_training_steps(self):
        """Samples from buffer and runs multiple training updates."""
        losses = []
        if not self.replay_buffer.can_sample():
            return losses

        for _ in range(self.args.num_train_steps_per_iteration):
            if not self.replay_buffer.can_sample():
                break # Not enough data for a full batch
            batch_data = self.replay_buffer.sample()
            loss = self._perform_update(batch_data)
            losses.append(loss)
        return losses

    @torch.no_grad() # No gradients needed for action selection
    def select_actions_epsilon_greedy(self, spatial_obs_list_np, vector_obs_list_np, 
                                      hidden_states_list_torch, avail_actions_list_np, 
                                      current_timestep_for_epsilon):
        """
        Selects actions for a batch of environments using epsilon-greedy.
        Inputs are lists, one element per environment.
        spatial_obs_list_np: list of [N, C, H, W] np.arrays
        vector_obs_list_np: list of [N, D_vec] np.arrays
        hidden_states_list_torch: list of [N, rnn_hidden_dim] torch tensors (on device)
        avail_actions_list_np: list of [N, n_actions] boolean np.arrays
        current_timestep_for_epsilon: global timestep for annealing epsilon.
        Returns: 
            chosen_actions_per_env (list of np.arrays [N] of action_indices), 
            new_hidden_states_per_env (list of torch tensors [N, rnn_hidden_dim]),
            epsilon (float)
        """
        num_active_envs = len(spatial_obs_list_np)
        if num_active_envs == 0:
            return [], [], 0.0 # Should not happen if called correctly

        # --- Batch observations for network input ---
        # Stack observations from all active environments:
        # spatial_obs_tensor: (num_active_envs * N, C, H, W)
        # vector_obs_tensor: (num_active_envs * N, D_vec)
        # hidden_states_tensor: (num_active_envs * N, rnn_hidden_dim)
        # avail_actions_tensor: (num_active_envs * N, n_actions)
        spatial_obs_tensor = torch.from_numpy(np.concatenate(spatial_obs_list_np, axis=0)).float().to(self.device)
        vector_obs_tensor = torch.from_numpy(np.concatenate(vector_obs_list_np, axis=0)).float().to(self.device)
        hidden_states_tensor = torch.cat(hidden_states_list_torch, dim=0).to(self.device) # Already tensors
        avail_actions_tensor = torch.from_numpy(np.concatenate(avail_actions_list_np, axis=0)).bool().to(self.device)

        self.agent_net.eval() # Set to evaluation mode
        q_values_batched, new_hidden_states_batched = self.agent_net(spatial_obs_tensor, vector_obs_tensor, hidden_states_tensor)
        self.agent_net.train() # Back to train mode

        # --- Epsilon-greedy selection ---
        # Anneal epsilon (linear decay)
        epsilon = self.args.epsilon_finish + (self.args.epsilon_start - self.args.epsilon_finish) * \
                  max(0., (self.args.epsilon_anneal_time - current_timestep_for_epsilon) / self.args.epsilon_anneal_time)
        if not self.args.use_epsilon_greedy: # If disabled, always act greedily
            epsilon = 0.0

        # Mask unavailable actions before taking argmax or random choice
        q_values_batched_masked = q_values_batched.clone() # Avoid modifying original q_values
        q_values_batched_masked[~avail_actions_tensor] = -float('inf')

        # Greedy actions
        greedy_actions = q_values_batched_masked.argmax(dim=1) # (num_active_envs * N)

        # Random actions (must be chosen from available actions)
        random_actions = torch.zeros_like(greedy_actions)
        for i in range(greedy_actions.shape[0]): # Iterate over each agent in the batch
            avail_idx_for_agent = torch.where(avail_actions_tensor[i])[0]
            if len(avail_idx_for_agent) > 0:
                random_actions[i] = avail_idx_for_agent[torch.randint(0, len(avail_idx_for_agent), (1,)).item()]
            else: 
                # This case (no available actions) should ideally be handled by the environment
                # or by ensuring at least one action (e.g., 'stay') is always available.
                # Fallback to the first action if no available actions.
                random_actions[i] = 0 

        # Choose based on epsilon
        chose_random = (torch.rand(greedy_actions.shape[0], device=self.device) < epsilon)
        chosen_actions_flat = torch.where(chose_random, random_actions, greedy_actions).cpu().numpy() # (num_active_envs * N)

        # --- Reshape actions and hidden states back to per-environment lists ---
        chosen_actions_per_env = [
            chosen_actions_flat[i*self.n_agents : (i+1)*self.n_agents] for i in range(num_active_envs)
        ]
        
        new_hidden_states_batched_cpu = new_hidden_states_batched.cpu()
        new_hidden_states_per_env = [
            new_hidden_states_batched_cpu[i*self.n_agents : (i+1)*self.n_agents] for i in range(num_active_envs)
        ]
        
        return chosen_actions_per_env, new_hidden_states_per_env, epsilon

    def collect_data_iteration(self, current_global_timestep_for_epsilon):
        """Collects one batch of episodes from parallel environments and stores them."""
        timesteps_collected_this_iter = 0
        episodes_info_this_iter = [] # List of dicts {'reward': r, 'length': l}
        epsilon_for_logging = None

        # Reset hidden states at the start of each new data collection round
        hidden_init_template = self.agent_net.init_hidden() # Expected (1, rnn_hidden_dim)
        if hidden_init_template.dim() == 3 and hidden_init_template.shape[0] == 1:
            # This case handles if init_hidden() unexpectedly returns (1, 1, H)
            hidden_init_template = hidden_init_template.squeeze(0) # -> (1, H)
        elif hidden_init_template.dim() == 2 and hidden_init_template.shape[0] == 1:
            # This is the expected case: (1, H)
            pass # hidden_init_template is already (1, H)
        # else: Could add a warning or error if shape is unexpected

        self.current_agent_hidden_states_list = [
            hidden_init_template.expand(self.n_agents, -1).to(self.device) # (1,H) -> (N,H)
            for _ in range(self.args.num_parallel_envs)
        ]
        
        current_episode_transitions_batch_for_buffer = [[] for _ in range(self.args.num_parallel_envs)]
        current_episode_rewards_this_iteration = np.zeros(self.args.num_parallel_envs)
        current_episode_lengths_this_iteration = np.zeros(self.args.num_parallel_envs, dtype=int)
        
        prev_env_states_list_for_transition = [s.copy() for s in self.current_env_states_list]
        prev_persistent_packages_list_for_transition = [{k:v.copy() for k,v in p.items()} for p in self.persistent_packages_list]
        
        active_envs_mask = [True] * self.args.num_parallel_envs
        
        for t_step_in_episode in range(self.args.episode_limit):
            if self.args.render:
                self.vec_env.render()
            if not any(active_envs_mask):
                break

            spatial_obs_for_net_active_envs = []
            vector_obs_for_net_active_envs = []
            avail_actions_for_net_active_envs = []
            hidden_states_for_net_active_envs = []
            env_indices_still_active_this_step = []

            for env_idx in range(self.args.num_parallel_envs):
                if not active_envs_mask[env_idx]:
                    continue
                env_indices_still_active_this_step.append(env_idx)

                env_state_dict = self.current_env_states_list[env_idx]
                persistent_pkgs_this_env = self.persistent_packages_list[env_idx]
                
                spatial_obs_all_agents_this_env = []
                vector_obs_all_agents_this_env = []
                avail_actions_all_agents_this_env = []

                for agent_id in range(self.args.n_agents):
                    so = convert_observation(env_state_dict, persistent_pkgs_this_env, agent_id)
                    vo = generate_vector_features(env_state_dict, persistent_pkgs_this_env, agent_id,
                                                  self.args.max_time_steps_env, self.args.max_other_robots_to_observe,
                                                  self.args.max_packages_to_observe)
                    spatial_obs_all_agents_this_env.append(so)
                    vector_obs_all_agents_this_env.append(vo)
                    
                    # Directly assume all actions are available based on user info and to fix AttributeError
                    avail_ac = np.ones(self.action_dim, dtype=bool)
                    avail_actions_all_agents_this_env.append(avail_ac)


                spatial_obs_for_net_active_envs.append(np.stack(spatial_obs_all_agents_this_env))
                vector_obs_for_net_active_envs.append(np.stack(vector_obs_all_agents_this_env))
                avail_actions_for_net_active_envs.append(np.stack(avail_actions_all_agents_this_env))
                hidden_states_for_net_active_envs.append(self.current_agent_hidden_states_list[env_idx])

            if not spatial_obs_for_net_active_envs:
                break

            chosen_actions_int_list_active, next_hidden_states_list_active, current_epsilon = \
                self.select_actions_epsilon_greedy(
                    spatial_obs_for_net_active_envs,  
                    vector_obs_for_net_active_envs,   
                    hidden_states_for_net_active_envs,
                    avail_actions_for_net_active_envs,
                    current_global_timestep_for_epsilon 
                )
            if t_step_in_episode == 0: # Log epsilon once per collection iteration
                 epsilon_for_logging = current_epsilon

            env_actions_list_for_step = []
            action_indices_for_buffer_active = [] 

            for actions_for_one_active_env in chosen_actions_int_list_active:
                env_actions_this_active_env = []
                for agent_action_int in actions_for_one_active_env:
                    move_idx = agent_action_int % self.args.num_move_actions
                    pkg_op_idx = agent_action_int // self.args.num_move_actions
                    if pkg_op_idx >= self.args.num_pkg_ops: pkg_op_idx = 0

                    move_str = self.le_move.inverse_transform([move_idx])[0]
                    pkg_op_str = self.le_pkg_op.inverse_transform([pkg_op_idx])[0]
                    env_actions_this_active_env.append((move_str, pkg_op_str))
                env_actions_list_for_step.append(env_actions_this_active_env)
                action_indices_for_buffer_active.append(actions_for_one_active_env)

            results_list_from_step = self.vec_env.step(env_actions_list_for_step, env_indices_still_active_this_step)
            
            # Unpack the results as four lists
            next_env_states_list, global_rewards_list, terminated_list, infos_list = results_list_from_step

            active_env_counter_in_results = 0
            for original_env_idx in env_indices_still_active_this_step:
                next_env_state_dict = next_env_states_list[active_env_counter_in_results]
                global_reward = global_rewards_list[active_env_counter_in_results]
                terminated = terminated_list[active_env_counter_in_results]
                info = infos_list[active_env_counter_in_results]
                
                actions_taken_this_env = env_actions_list_for_step[active_env_counter_in_results]
                self._update_persistent_packages_for_env(original_env_idx, next_env_state_dict)

                # Fix: Extract reward if it's a dict
                current_episode_rewards_this_iteration[original_env_idx] += global_reward
                current_episode_lengths_this_iteration[original_env_idx] += 1
                timesteps_collected_this_iter += 1

                so_next_all_agents_this_env = []
                vo_next_all_agents_this_env = []
                avail_u_next_all_agents_this_env = []
                for agent_id in range(self.args.n_agents):
                    so_next = convert_observation(next_env_state_dict, self.persistent_packages_list[original_env_idx], agent_id)
                    vo_next = generate_vector_features(next_env_state_dict, self.persistent_packages_list[original_env_idx], agent_id,
                                                       self.args.max_time_steps_env, self.args.max_other_robots_to_observe,
                                                       self.args.max_packages_to_observe)
                    so_next_all_agents_this_env.append(so_next)
                    vo_next_all_agents_this_env.append(vo_next)
                    
                    # Directly assume all actions are available for next state as well
                    avail_next = np.ones(self.action_dim, dtype=bool)
                    avail_u_next_all_agents_this_env.append(avail_next)


                gs_s_next, gs_v_next = convert_global_state(next_env_state_dict, self.persistent_packages_list[original_env_idx],
                                                            self.args.max_time_steps_env, self.args.max_robots_in_state,
                                                            self.args.max_packages_in_state)
                
                gs_s_current, gs_v_current = convert_global_state(
                    prev_env_states_list_for_transition[original_env_idx], 
                    prev_persistent_packages_list_for_transition[original_env_idx],
                    self.args.max_time_steps_env, self.args.max_robots_in_state, self.args.max_packages_in_state
                )
                
                current_episode_transitions_batch_for_buffer[original_env_idx].append({
                    "so": spatial_obs_for_net_active_envs[active_env_counter_in_results],
                    "vo": vector_obs_for_net_active_envs[active_env_counter_in_results],
                    "gs": gs_s_current, "gv": gs_v_current,
                    "u": action_indices_for_buffer_active[active_env_counter_in_results],
                    "r": global_reward,
                    "so_next": np.stack(so_next_all_agents_this_env),
                    "vo_next": np.stack(vo_next_all_agents_this_env),
                    "gs_next": gs_s_next, "gv_next": gs_v_next,
                    "avail_u": avail_actions_for_net_active_envs[active_env_counter_in_results],
                    "avail_u_next": np.stack(avail_u_next_all_agents_this_env),
                    "terminated": terminated, "padded": False
                })

                self.current_env_states_list[original_env_idx] = next_env_state_dict
                self.current_agent_hidden_states_list[original_env_idx] = next_hidden_states_list_active[active_env_counter_in_results] 
                
                prev_env_states_list_for_transition[original_env_idx] = next_env_state_dict.copy()
                prev_persistent_packages_list_for_transition[original_env_idx] = {
                    k:v.copy() for k,v in self.persistent_packages_list[original_env_idx].items()
                }

                if terminated:
                    active_envs_mask[original_env_idx] = False
                    episodes_info_this_iter.append({
                        'reward': current_episode_rewards_this_iteration[original_env_idx],
                        'length': current_episode_lengths_this_iteration[original_env_idx]
                    })
                    if len(current_episode_transitions_batch_for_buffer[original_env_idx]) > 0:
                        self.replay_buffer.add_episode_data(current_episode_transitions_batch_for_buffer[original_env_idx])
                active_env_counter_in_results += 1

        for env_idx in range(self.args.num_parallel_envs):
            if active_envs_mask[env_idx] and len(current_episode_transitions_batch_for_buffer[env_idx]) > 0:
                episodes_info_this_iter.append({
                    'reward': current_episode_rewards_this_iteration[env_idx],
                    'length': current_episode_lengths_this_iteration[env_idx]
                })
                self.replay_buffer.add_episode_data(current_episode_transitions_batch_for_buffer[env_idx])
        
        # Prepare for the next data collection iteration by resetting environments and their persistent packages
        # The hidden states for the next iteration will be reset at the start of the next call to this method.
        self.current_env_states_list = self.vec_env.reset()
        self.persistent_packages_list = [{} for _ in range(self.args.num_parallel_envs)]
        for env_idx, state_dict in enumerate(self.current_env_states_list):
            self._update_persistent_packages_for_env(env_idx, state_dict)

        return timesteps_collected_this_iter, episodes_info_this_iter, epsilon_for_logging

    def save_models_learner(self, path_prefix, episode_count):
        save_qmix_model(self.agent_net, self.mixer_net, f"{path_prefix}_ep{episode_count}")

    def load_models_learner(self, path_prefix, episode_count):
        loaded = load_qmix_model(self.agent_net, self.mixer_net, f"{path_prefix}_ep{episode_count}", device=self.device)
        if loaded:
            self.update_target_networks(hard_update=True) # Sync target nets after loading
        return loaded

# --- Main Training Script ---
if __name__ == "__main__":
    args = Args() # Load hyperparameters
    print(f"Using device: {device}")
    # --- Environment and Action Conversion Setup ---
    MAP_FILE = args.map_file if hasattr(args, "map_file") else "QMIX/marl_delivery/map1.txt"
    N_PACKAGES = args.n_packages
    MOVE_COST = args.move_cost if hasattr(args, "move_cost") else -0.01
    DELIVERY_REWARD = args.delivery_reward if hasattr(args, "delivery_reward") else 10
    DELAY_REWARD = args.delay_reward if hasattr(args, "delay_reward") else 1
    NUM_AGENTS = args.n_agents
    SEED = args.seed if args.seed is not None else 42
    NUM_ENVS = args.num_parallel_envs
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)

    # Action conversion (environment uses string actions)
    le_move = LabelEncoder()
    le_move.fit(['S', 'L', 'R', 'U', 'D']) # Stay, Left, Right, Up, Down
    le_pkg_op = LabelEncoder()
    le_pkg_op.fit(['0', '1', '2']) # 0: None, 1: Pickup, 2: Drop
    args.num_move_actions = len(le_move.classes_)
    args.num_pkg_ops = len(le_pkg_op.classes_)
    args.action_dim = args.num_move_actions * args.num_pkg_ops # Total discrete actions

    # --- Determine Observation and State Shapes ---
    print("Initializing temporary environment to get observation/state shapes...")
    try:
        # Use a single instance of the environment to get shapes
        from env import Environment
        _temp_env = Environment(
            map_file=MAP_FILE,
            n_robots=args.n_agents,
            n_packages=args.n_packages,
            move_cost=MOVE_COST,
            delivery_reward=DELIVERY_REWARD,
            delay_reward=DELAY_REWARD,
            seed=SEED,
            max_time_steps=args.max_time_steps_env
        )
        temp_env_state_dict = _temp_env.reset() # Returns dict
        
        # Initialize persistent packages for the temp env for shape calculation
        temp_persistent_packages = {}

        _s_obs_example = convert_observation(temp_env_state_dict, temp_persistent_packages, 0)
        args.spatial_obs_shape = _s_obs_example.shape
        
        _v_obs_example = generate_vector_features(temp_env_state_dict, temp_persistent_packages, 0,
                                                  args.max_time_steps_env, args.max_other_robots_to_observe,
                                                  args.max_packages_to_observe)
        args.vector_obs_dim = _v_obs_example.shape[0]

        _gs_s_example, _gs_v_example = convert_global_state(temp_env_state_dict, temp_persistent_packages,
                                                            args.max_time_steps_env, args.max_robots_in_state,
                                                            args.max_packages_in_state)
        args.global_spatial_state_shape = _gs_s_example.shape
        args.global_vector_state_dim = _gs_v_example.shape[0]

        print(f"Spatial Obs Shape: {args.spatial_obs_shape}, Vector Obs Dim: {args.vector_obs_dim}")
        print(f"Global Spatial State Shape: {args.global_spatial_state_shape}, Global Vector State Dim: {args.global_vector_state_dim}")
        print(f"Action Dim: {args.action_dim}, Num Agents: {args.n_agents}")

    except ImportError as e:
        print(f"Could not import environment for shape determination: {e}")
        exit()
    except Exception as e:
        print(f"Error during temporary environment initialization or shape calculation: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Initialize Learner and Replay Buffer ---
    replay_buffer = ReplayBuffer(
        args.buffer_size, args.episode_limit, args.n_agents,
        args.spatial_obs_shape, args.vector_obs_dim,
        args.global_spatial_state_shape, args.global_vector_state_dim,
        args.action_dim, 
        args
    )

    # --- Vectorized Environment for Data Collection ---
    print(f"Initializing {NUM_ENVS} parallel environments...")
    try:
        vec_env = VectorizedEnv(
            Environment, num_envs=NUM_ENVS,
            map_file=MAP_FILE,
            n_robots=NUM_AGENTS,
            n_packages=args.n_packages,
            move_cost=MOVE_COST,
            delivery_reward=DELIVERY_REWARD,
            delay_reward=DELAY_REWARD,
            seed=SEED,
            max_time_steps=args.max_time_steps_env
        )
    except Exception as e:
        print(f"Failed to initialize VectorizedEnv: {e}. Ensure 'env.py' and 'env_vectorized.py' are correct.")
        exit()


    try:
        # --- Initialize QMixLearner (New way) ---
        qmix_trainer = QMixLearner(
            n_agents=args.n_agents,
            action_dim=args.action_dim,
            args=args,
            vec_env=vec_env,
            replay_buffer=replay_buffer,
            le_move=le_move,
            le_pkg_op=le_pkg_op
        )

        # --- Training Loop ---
        total_timesteps_collected = 0
        total_episodes_collected = 0
        episode_rewards_history = []
        episode_lengths_history = []
        losses_history = []
        epsilon_history_log = [] # For logging epsilon at each action selection

        # Load model if specified
        if args.load_model_path and args.load_model_episode > 0:
            print(f"Loading model from {args.load_model_path} at episode {args.load_model_episode}...")
            qmix_trainer.load_models_learner(args.load_model_path, args.load_model_episode)

        print(f"Starting QMIX training on {device}...")

        qmix_trainer._initialize_collection_state(initial_reset=True) # Initial reset of envs and internal states

        # Main training iterations (each iteration collects data and potentially trains)
        for training_iteration in range(1, args.max_training_iterations + 1):
            
            # --- Data Collection Phase ---
            # Hidden states are reset inside collect_data_iteration for the new batch of episodes
            iter_timesteps, iter_episodes_info, iter_epsilon = \
                qmix_trainer.collect_data_iteration(total_timesteps_collected)

            total_timesteps_collected += iter_timesteps
            for ep_info in iter_episodes_info:
                total_episodes_collected += 1
                episode_rewards_history.append(ep_info['reward'])
                episode_lengths_history.append(ep_info['length'])
            
            if iter_epsilon is not None and training_iteration % args.log_interval == 0 :
                epsilon_history_log.append(iter_epsilon)


            # --- Training Phase ---
            if qmix_trainer.replay_buffer.can_sample() and \
            total_timesteps_collected > args.min_timesteps_to_train : # Ensure some initial exploration
                
                iter_losses = qmix_trainer.run_training_steps()
                losses_history.extend(iter_losses)

            # --- Logging and Saving ---
            if training_iteration % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards_history[-args.log_interval*NUM_ENVS:]) if episode_rewards_history else 0
                avg_length = np.mean(episode_lengths_history[-args.log_interval*NUM_ENVS:]) if episode_lengths_history else 0
                avg_loss = np.mean(losses_history[-args.log_interval*NUM_ENVS*args.num_train_steps_per_iteration:]) if losses_history else 0 # Approx
                last_epsilon = epsilon_history_log[-1] if epsilon_history_log else args.epsilon_start
                
                print(f"Iter: {training_iteration}/{args.max_training_iterations} | Total Timesteps: {total_timesteps_collected} | Total Episodes: {total_episodes_collected}")
                print(f"  Avg Reward (last {args.log_interval*NUM_ENVS} eps): {avg_reward:.2f} | Avg Length: {avg_length:.2f}")
                print(f"  Avg Loss (approx): {avg_loss:.4f} | Epsilon: {last_epsilon:.3f}")
                print(f"  Buffer Size: {len(qmix_trainer.replay_buffer)}/{args.buffer_size}")

            if training_iteration > 0 and training_iteration % args.save_model_interval == 0:
                print(f"Saving model at iteration {training_iteration}, episode {total_episodes_collected}...")
                qmix_trainer.save_models_learner(f"models/qmix_iter{training_iteration}", total_episodes_collected)

            if total_timesteps_collected >= args.max_total_timesteps:
                print(f"Reached max training timesteps: {args.max_total_timesteps}.")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        # --- End of Training Loop ---
        print("Training finished.")
        qmix_trainer.save_models_learner("models/qmix_final", total_episodes_collected)

        # --- Plotting ---
        if not os.path.exists("plots"): os.makedirs("plots")
        plt.figure(figsize=(18, 10))
        plt.subplot(2, 2, 1)
        if episode_rewards_history:
            plt.plot(episode_rewards_history)
            # Moving average
            if len(episode_rewards_history) >= 100:
                rewards_moving_avg = np.convolve(episode_rewards_history, np.ones(100)/100, mode='valid')
                plt.plot(np.arange(99, len(episode_rewards_history)), rewards_moving_avg, label='100-ep MA')
                plt.legend()
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        if losses_history:
            plt.plot(losses_history)
        plt.title('QMIX Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        if episode_lengths_history:
            plt.plot(episode_lengths_history)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        if epsilon_history_log: 
            plt.plot(epsilon_history_log)
            plt.title('Epsilon Decay')
            plt.xlabel(f'Logged every {args.log_interval} iterations')
            plt.ylabel('Epsilon')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig("plots/qmix_training_plots.png")
        print("Training plots saved to plots/qmix_training_plots.png")
        plt.show()