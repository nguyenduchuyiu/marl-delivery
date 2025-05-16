# trainer.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from networks import ActorNetwork, CriticNetwork
from MAPPO.helper import (
    compute_shaped_rewards,
    convert_global_state,
    convert_observation,
    generate_vector_features,
    save_mappo_model,
    load_mappo_model
)


from env import Environment # Not strictly needed if only using VectorizedEnv for training
from env_vectorized import VectorizedEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder # For action conversion
import matplotlib.pyplot as plt

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- MAPPO Hyperparameters ---
ACTION_DIM = 15  # Total discrete actions for an agent
NUM_AGENTS = 5
MAP_FILE = "marl_delivery/map1.txt"
N_PACKAGES = 50
MOVE_COST = -0.01 # Adjusted for PPO, rewards should be reasonably scaled
DELIVERY_REWARD = 10
DELAY_REWARD = 1 # Or 0, depending on reward shaping strategy
MAX_TIME_STEPS_PER_EPISODE = 500 # Max steps for one episode in one env

NUM_ENVS = 5  # Number of parallel environments
ROLLOUT_STEPS = 500 # Number of steps to collect data for before an update
TOTAL_TIMESTEPS = 1_000_000 # Total timesteps for training

# PPO specific
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
NUM_EPOCHS = 5 # Number of epochs to train on collected data
MINIBATCH_SIZE = 64 # Minibatch size for PPO updates
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
WEIGHT_DECAY = 1e-4

class MAPPOTrainer:
    def __init__(self, vec_env, num_agents, action_dim, obs_shape, global_state_shape,
                 vector_obs_dim, global_vector_state_dim):
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.obs_shape = obs_shape # (C, H, W) for local obs
        self.global_state_shape = global_state_shape # (C_global, H, W) for global state
        self.vector_obs_dim = vector_obs_dim
        self.global_vector_state_dim = global_vector_state_dim

        self.actor = ActorNetwork(obs_shape, vector_obs_dim, action_dim).to(device)
        self.critic = CriticNetwork(global_state_shape, global_vector_state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # For converting integer actions to environment actions
        self.le_move = LabelEncoder()
        self.le_move.fit(['S', 'L', 'R', 'U', 'D'])
        self.le_pkg_op = LabelEncoder()
        self.le_pkg_op.fit(['0', '1', '2']) # 0: None, 1: Pickup, 2: Drop
        self.NUM_MOVE_ACTIONS = len(self.le_move.classes_)
        self.NUM_PKG_OPS = len(self.le_pkg_op.classes_)
        
        # Persistent packages trackers for each environment (for state conversion)
        self.persistent_packages_list = [{} for _ in range(self.num_envs)]


    def _update_persistent_packages_for_env(self, env_idx, current_env_state_dict):
        # This is a simplified version of the DQNTrainer's method, adapted for one env
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


    def _get_actions_and_values(self, current_local_obs_b_a_c_h_w, current_vector_obs_b_a_d, current_global_states_b_c_h_w, current_global_vector_b_d):
        # Ensure input tensors are on the correct device
        current_local_obs_b_a_c_h_w = current_local_obs_b_a_c_h_w.to(device)
        current_vector_obs_b_a_d = current_vector_obs_b_a_d.to(device)
        current_global_states_b_c_h_w = current_global_states_b_c_h_w.to(device)
        current_global_vector_b_d = current_global_vector_b_d.to(device)
        
        actor_input_obs = current_local_obs_b_a_c_h_w.reshape(self.num_envs * self.num_agents, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        actor_input_vec = current_vector_obs_b_a_d.reshape(self.num_envs * self.num_agents, self.vector_obs_dim)
        action_logits = self.actor(actor_input_obs, actor_input_vec) # (NUM_ENVS * NUM_AGENTS, ACTION_DIM)
        dist = Categorical(logits=action_logits)
        actions_int = dist.sample() # (NUM_ENVS * NUM_AGENTS)
        log_probs = dist.log_prob(actions_int) # (NUM_ENVS * NUM_AGENTS)

        actions_int_reshaped = actions_int.reshape(self.num_envs, self.num_agents)
        log_probs_reshaped = log_probs.reshape(self.num_envs, self.num_agents)

        values = self.critic(current_global_states_b_c_h_w, current_global_vector_b_d) # (NUM_ENVS, 1)

        return actions_int_reshaped, log_probs_reshaped, values.squeeze(-1) # values squeezed to (NUM_ENVS)

    def collect_rollouts(self, current_env_states_list, current_local_obs_list, current_vector_obs_list, current_global_states_list, current_global_vector_list):
        # Buffers to store trajectory data
        mb_obs = torch.zeros((ROLLOUT_STEPS, self.num_envs, self.num_agents, *self.obs_shape), device=device)
        mb_vector_obs = torch.zeros((ROLLOUT_STEPS, self.num_envs, self.num_agents, self.vector_obs_dim), device=device)
        mb_global_states = torch.zeros((ROLLOUT_STEPS, self.num_envs, *self.global_state_shape), device=device)
        mb_global_vector = torch.zeros((ROLLOUT_STEPS, self.num_envs, self.global_vector_state_dim), device=device)
        mb_actions = torch.zeros((ROLLOUT_STEPS, self.num_envs, self.num_agents), dtype=torch.long, device=device)
        mb_log_probs = torch.zeros((ROLLOUT_STEPS, self.num_envs, self.num_agents), device=device)
        mb_rewards = torch.zeros((ROLLOUT_STEPS, self.num_envs), device=device)
        mb_dones = torch.zeros((ROLLOUT_STEPS, self.num_envs), dtype=torch.bool, device=device)
        mb_values = torch.zeros((ROLLOUT_STEPS, self.num_envs), device=device)

        # Move initial obs/states to device
        current_local_obs_list = current_local_obs_list.to(device)
        current_vector_obs_list = current_vector_obs_list.to(device)
        current_global_states_list = current_global_states_list.to(device)
        current_global_vector_list = current_global_vector_list.to(device)

        for step in range(ROLLOUT_STEPS):
            # Render the environment
            self.vec_env.render()
            
            mb_obs[step] = current_local_obs_list
            mb_vector_obs[step] = current_vector_obs_list
            mb_global_states[step] = current_global_states_list
            mb_global_vector[step] = current_global_vector_list

            with torch.no_grad():
                actions_int_ne_na, log_probs_ne_na, values_ne = self._get_actions_and_values(
                    current_local_obs_list, 
                    current_vector_obs_list,
                    current_global_states_list,
                    current_global_vector_list
                )
            
            mb_actions[step] = actions_int_ne_na
            mb_log_probs[step] = log_probs_ne_na
            mb_values[step] = values_ne

            # Convert integer actions to environment compatible actions
            env_actions_batch = []
            for env_idx in range(self.num_envs):
                env_agent_actions = []
                for agent_idx in range(self.num_agents):
                    int_act = actions_int_ne_na[env_idx, agent_idx].item()
                    move_idx = int_act % self.NUM_MOVE_ACTIONS
                    pkg_op_idx = int_act // self.NUM_MOVE_ACTIONS
                    if pkg_op_idx >= self.NUM_PKG_OPS: pkg_op_idx = 0 # Safety clamp
                    
                    move_str = self.le_move.inverse_transform([move_idx])[0]
                    pkg_op_str = self.le_pkg_op.inverse_transform([pkg_op_idx])[0]
                    env_agent_actions.append((move_str, pkg_op_str))
                env_actions_batch.append(env_agent_actions)

            next_env_states_list, global_rewards_ne, dones_ne, _ = self.vec_env.step(env_actions_batch)
            
            # use reward shaping here
            reshaped_global_rewards_ne = [compute_shaped_rewards(
                                                        global_rewards_ne[env_idx],
                                                        current_env_states_list[env_idx], 
                                                        next_env_states_list[env_idx], 
                                                        env_actions_batch[env_idx], 
                                                        self.persistent_packages_list[env_idx],
                                                        self.num_agents) 
                                          for env_idx in range(self.num_envs)]
            
            mb_rewards[step] = torch.tensor(reshaped_global_rewards_ne, dtype=torch.float32, device=device)
            mb_dones[step] = torch.tensor(dones_ne, dtype=torch.bool, device=device)

            # Prepare next observations and states
            next_local_obs_list = torch.zeros_like(current_local_obs_list, device=device)
            next_vector_obs_list = torch.zeros_like(current_vector_obs_list, device=device)
            next_global_states_list = torch.zeros_like(current_global_states_list, device=device)
            next_global_vector_list = torch.zeros_like(current_global_vector_list, device=device)

            for env_idx in range(self.num_envs):
                if dones_ne[env_idx]:
                    # --- Reset environment if done ---
                    reset_state = self.vec_env.envs[env_idx].reset()
                    self._update_persistent_packages_for_env(env_idx, reset_state)
                    current_persistent_packages = self.persistent_packages_list[env_idx]
                    next_env_states_list[env_idx] = reset_state
                    # Update global state and local obs after reset
                    next_global_states_list[env_idx] = torch.from_numpy(convert_global_state(
                            reset_state, 
                            current_persistent_packages, 
                            MAX_TIME_STEPS_PER_EPISODE,
                        )[0]
                    ).to(device)
                    next_global_vector_list[env_idx] = torch.from_numpy(
                        convert_global_state(reset_state, 
                                             current_persistent_packages, 
                                             MAX_TIME_STEPS_PER_EPISODE
                                             )[1]
                    ).float().to(device)
                    
                    for agent_idx in range(self.num_agents):
                        next_local_obs_list[env_idx, agent_idx] = torch.from_numpy(
                            convert_observation(reset_state, current_persistent_packages, agent_idx)
                        ).float().to(device)
                        next_vector_obs_list[env_idx, agent_idx] = torch.from_numpy(
                            generate_vector_features(reset_state, current_persistent_packages, agent_idx,
                                                    MAX_TIME_STEPS_PER_EPISODE, self.num_agents-1, 5)
                        ).float().to(device)
                else:
                    self._update_persistent_packages_for_env(env_idx, next_env_states_list[env_idx])
                    current_persistent_packages = self.persistent_packages_list[env_idx]
                    next_global_states_list[env_idx] = torch.from_numpy(convert_global_state(
                            next_env_states_list[env_idx], 
                            current_persistent_packages, 
                            MAX_TIME_STEPS_PER_EPISODE,
                        )[0]
                    ).to(device)
                    next_global_vector_list[env_idx] = torch.from_numpy(
                        convert_global_state(next_env_states_list[env_idx], 
                                             current_persistent_packages, 
                                             MAX_TIME_STEPS_PER_EPISODE
                                             )[1]
                    ).float().to(device)
                    for agent_idx in range(self.num_agents):
                        next_local_obs_list[env_idx, agent_idx] = torch.from_numpy(
                            convert_observation(next_env_states_list[env_idx], current_persistent_packages, agent_idx)
                        ).float().to(device)
                        next_vector_obs_list[env_idx, agent_idx] = torch.from_numpy(
                            generate_vector_features(next_env_states_list[env_idx], current_persistent_packages, agent_idx,
                                                    MAX_TIME_STEPS_PER_EPISODE, self.num_agents-1, 5)
                        ).float().to(device)
            
            current_env_states_list = next_env_states_list
            current_local_obs_list = next_local_obs_list
            current_vector_obs_list = next_vector_obs_list
            current_global_states_list = next_global_states_list
            current_global_vector_list = next_global_vector_list
        
        # Calculate advantages using GAE
        advantages = torch.zeros_like(mb_rewards, device=device)
        last_gae_lambda = 0
        with torch.no_grad():
            # Get value of the last state in the rollout
            next_value_ne = self.critic(current_global_states_list, current_global_vector_list).squeeze(-1) # (NUM_ENVS)

        for t in reversed(range(ROLLOUT_STEPS)):
            next_non_terminal = 1.0 - mb_dones[t].float()
            next_values_step = next_value_ne if t == ROLLOUT_STEPS - 1 else mb_values[t+1]
            
            delta = mb_rewards[t] + GAMMA * next_values_step * next_non_terminal - mb_values[t]
            advantages[t] = last_gae_lambda = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lambda
        
        returns = advantages + mb_values

        # Flatten the batch for training
        b_obs = mb_obs.reshape(-1, *self.obs_shape)
        b_vector_obs = mb_vector_obs.reshape(-1, self.vector_obs_dim)
        b_global_states = mb_global_states.reshape(ROLLOUT_STEPS * self.num_envs, *self.global_state_shape)
        b_global_vector = mb_global_vector.reshape(ROLLOUT_STEPS * self.num_envs, self.global_vector_state_dim)
        b_actions = mb_actions.reshape(-1)
        b_log_probs = mb_log_probs.reshape(-1)
        
        b_advantages = advantages.reshape(ROLLOUT_STEPS * self.num_envs, 1).repeat(1, self.num_agents).reshape(-1)
        b_returns_critic = returns.reshape(-1)

        return (b_obs, b_vector_obs, b_global_states, b_global_vector, b_actions, 
           b_log_probs, b_advantages, b_returns_critic,
           current_env_states_list, current_local_obs_list, current_vector_obs_list, current_global_states_list, current_global_vector_list,
           mb_rewards)

    def update_ppo(self, b_obs, b_vector_obs, b_global_states, b_global_vector, b_actions, b_log_probs_old, b_advantages, b_returns_critic):
        # Ensure all tensors are on the correct device
        b_obs = b_obs.to(device)
        b_vector_obs = b_vector_obs.to(device)
        b_global_states = b_global_states.to(device)
        b_global_vector = b_global_vector.to(device)
        b_actions = b_actions.to(device)
        b_log_probs_old = b_log_probs_old.to(device)
        b_advantages = b_advantages.to(device)
        b_returns_critic = b_returns_critic.to(device)

        num_samples_actor = ROLLOUT_STEPS * self.num_envs * self.num_agents
        num_samples_critic = ROLLOUT_STEPS * self.num_envs
        
        actor_batch_indices = np.arange(num_samples_actor)
        critic_batch_indices = np.arange(num_samples_critic)

        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(actor_batch_indices)
            np.random.shuffle(critic_batch_indices)

            # Actor update
            for start in range(0, num_samples_actor, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_indices = actor_batch_indices[start:end]

                mb_obs_slice = b_obs[mb_indices]
                mb_vector_obs_slice = b_vector_obs[mb_indices]
                mb_actions_slice = b_actions[mb_indices]
                mb_log_probs_old_slice = b_log_probs_old[mb_indices]
                mb_advantages_slice = b_advantages[mb_indices]
                
                # Normalize advantages (optional but often helpful)
                mb_advantages_slice = (mb_advantages_slice - mb_advantages_slice.mean()) / (mb_advantages_slice.std() + 1e-8)

                action_logits = self.actor(mb_obs_slice, mb_vector_obs_slice)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(mb_actions_slice)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_log_probs_old_slice
                ratio = torch.exp(log_ratio)

                pg_loss1 = -mb_advantages_slice * ratio
                pg_loss2 = -mb_advantages_slice * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                total_actor_loss = actor_loss - ENTROPY_COEF * entropy
                
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optimizer.step()

            # Critic update
            for start in range(0, num_samples_critic, MINIBATCH_SIZE // self.num_agents if self.num_agents > 0 else MINIBATCH_SIZE):
                end = start + (MINIBATCH_SIZE // self.num_agents if self.num_agents > 0 else MINIBATCH_SIZE)
                mb_indices = critic_batch_indices[start:end]
                
                mb_global_states_slice = b_global_states[mb_indices]
                mb_global_vector_slice = b_global_vector[mb_indices]
                mb_returns_critic_slice = b_returns_critic[mb_indices]

                new_values = self.critic(mb_global_states_slice, mb_global_vector_slice).squeeze(-1)
                critic_loss = F.mse_loss(new_values, mb_returns_critic_slice)
                
                total_critic_loss = VALUE_LOSS_COEF * critic_loss

                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy.item()


if __name__ == "__main__":
    print(f"Using device: {device}")
    vec_env = VectorizedEnv(
        Environment, num_envs=NUM_ENVS,
        map_file=MAP_FILE,
        n_robots=NUM_AGENTS,
        n_packages=N_PACKAGES,
        move_cost=MOVE_COST,
        delivery_reward=DELIVERY_REWARD,
        delay_reward=DELAY_REWARD,
        seed=SEED, # Seed for each sub-environment will be SEED, SEED+1, ...
        max_time_steps=MAX_TIME_STEPS_PER_EPISODE
    )

    # Determine observation and global state shapes from one env instance
    _temp_env = Environment(map_file=MAP_FILE, n_robots=NUM_AGENTS, n_packages=N_PACKAGES, move_cost=MOVE_COST, delivery_reward=DELIVERY_REWARD, delay_reward=DELAY_REWARD, seed=SEED, max_time_steps=MAX_TIME_STEPS_PER_EPISODE)

            
    OBS_SHAPE = (6, _temp_env.n_rows, _temp_env.n_cols)
    GLOBAL_STATE_SHAPE = (4, _temp_env.n_rows, _temp_env.n_cols)
    VECTOR_OBS_DIM = generate_vector_features(_temp_env.reset(), {}, 0, MAX_TIME_STEPS_PER_EPISODE, NUM_AGENTS-1, 5).shape[0]
    GLOBAL_VECTOR_STATE_DIM = convert_global_state(_temp_env.reset(), {}, MAX_TIME_STEPS_PER_EPISODE)[1].shape[0]

    print(f"Obs shape: {OBS_SHAPE}, Global state shape: {GLOBAL_STATE_SHAPE}, Vector obs dim: {VECTOR_OBS_DIM}, Global vector dim: {GLOBAL_VECTOR_STATE_DIM}")

    trainer = MAPPOTrainer(vec_env, NUM_AGENTS, ACTION_DIM, OBS_SHAPE, GLOBAL_STATE_SHAPE, VECTOR_OBS_DIM, GLOBAL_VECTOR_STATE_DIM)

    # Load existing model if available
    load_mappo_model(trainer.actor, trainer.critic, device=device) # Uncomment to load

    episode_rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []
    rollout_reward_history = []

    print("Starting MAPPO training...")

    # Initial reset and state preparation
    current_env_states_list = vec_env.reset() # List of state dicts
    current_local_obs_list = torch.zeros((NUM_ENVS, NUM_AGENTS, *OBS_SHAPE), device="cpu")
    current_vector_obs_list = torch.zeros((NUM_ENVS, NUM_AGENTS, VECTOR_OBS_DIM), device="cpu")
    current_global_states_list = torch.zeros((NUM_ENVS, *GLOBAL_STATE_SHAPE), device="cpu")
    current_global_vector_list = torch.zeros((NUM_ENVS, GLOBAL_VECTOR_STATE_DIM), device="cpu")

    for env_idx in range(NUM_ENVS):
        trainer._update_persistent_packages_for_env(env_idx, current_env_states_list[env_idx])
        current_persistent_packages = trainer.persistent_packages_list[env_idx]
        current_global_states_list[env_idx] = torch.from_numpy(convert_global_state(
                                                            current_env_states_list[env_idx], 
                                                            current_persistent_packages, 
                                                            MAX_TIME_STEPS_PER_EPISODE,
                                                            )[0]
                                                            )
        current_global_vector_list[env_idx] = torch.from_numpy(convert_global_state(
                                                            current_env_states_list[env_idx], 
                                                            current_persistent_packages, 
                                                            MAX_TIME_STEPS_PER_EPISODE,
                                                            )[1]
                                                            )
        for agent_idx in range(NUM_AGENTS):
            current_local_obs_list[env_idx, agent_idx] = torch.from_numpy(
                convert_observation(current_env_states_list[env_idx], current_persistent_packages, agent_idx)
            ).float()
            current_vector_obs_list[env_idx, agent_idx] = torch.from_numpy(
                generate_vector_features(current_env_states_list[env_idx], current_persistent_packages, agent_idx,
                                        MAX_TIME_STEPS_PER_EPISODE, NUM_AGENTS-1, 5)
            ).float()

    num_updates = TOTAL_TIMESTEPS // (ROLLOUT_STEPS * NUM_ENVS)
    total_steps_done = 0

    try:
        for update_num in range(1, num_updates + 1):
            (b_obs, b_vector_obs, b_global_states, b_global_vector, b_actions, b_log_probs_old, 
                b_advantages, b_returns_critic,
                current_env_states_list, current_local_obs_list, current_vector_obs_list, current_global_states_list, current_global_vector_list,
                mb_rewards
            ) = trainer.collect_rollouts(current_env_states_list, current_local_obs_list, current_vector_obs_list, current_global_states_list, current_global_vector_list)
            
            # Track reward per rollout
            rollout_total_reward = mb_rewards.sum().item()
            rollout_reward_history.append(rollout_total_reward)

            print(f"Rollout {update_num}: Total Reward = {rollout_total_reward:.2f}")

            actor_loss, critic_loss, entropy = trainer.update_ppo(
                b_obs, b_vector_obs, b_global_states, b_global_vector, b_actions, b_log_probs_old, b_advantages, b_returns_critic
            )
            
            total_steps_done += ROLLOUT_STEPS * NUM_ENVS
            
            actor_loss_history.append(actor_loss)
            critic_loss_history.append(critic_loss)
            entropy_history.append(entropy)

            if update_num % 10 == 0: # Log every 10 updates
                print(f"Update {update_num}/{num_updates} | Timesteps: {total_steps_done}/{TOTAL_TIMESTEPS}")
                print(f"  Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | Entropy: {entropy:.4f}")
                print(f"  Rollout Total Reward: {rollout_total_reward:.2f}")

            if update_num % 100 == 0: # Save model periodically
                print(f"Saving checkpoint at update {update_num}...")
                save_mappo_model(trainer.actor, trainer.critic, path_prefix=f"models/mappo_update{update_num}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Saving final model...")
        save_mappo_model(trainer.actor, trainer.critic, path_prefix="models/mappo_final")
        print("\nTraining loop finished or was interrupted.")

        # Plotting
        plt.figure(figsize=(24, 5))
        plt.subplot(1, 4, 1)
        plt.plot(actor_loss_history)
        plt.title('Actor Loss per Update')
        plt.xlabel('Update Number')
        plt.ylabel('Actor Loss')
        plt.grid(True)

        plt.subplot(1, 4, 2)
        plt.plot(critic_loss_history)
        plt.title('Critic Loss per Update')
        plt.xlabel('Update Number')
        plt.ylabel('Critic Loss')
        plt.grid(True)
        
        plt.subplot(1, 4, 3)
        plt.plot(entropy_history)
        plt.title('Policy Entropy per Update')
        plt.xlabel('Update Number')
        plt.ylabel('Entropy')
        plt.grid(True)

        plt.subplot(1, 4, 4)
        plt.plot(rollout_reward_history)
        plt.title('Total Reward per Rollout')
        plt.xlabel('Update Number')
        plt.ylabel('Total Reward')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
