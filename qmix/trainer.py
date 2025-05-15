#trainer.py
from networks import AgentNetwork, MixingNetwork, ReplayBuffer, convert_state, convert_global_state_to_tensor
from env import Environment
from env_vectorized import VectorizedEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
import os
from sklearn.calibration import LabelEncoder
import matplotlib.pyplot as plt
import copy


SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

ACTION_DIM = 15
NUM_AGENTS = 5
MAP_FILE = "map1.txt"
N_PACKAGES = 20

MOVE_COST = -0.01 
DELIVERY_REWARD = 10
DELAY_REWARD = 1

MAX_TIME_STEPS = 1000
MIXING_DIM = 64
NUM_EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-5
WEIGHT_DECAY = 1e-3
MAX_REPLAY_BUFFER_SIZE = 50000
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.001
GRADIENT_CLIPPING = 10

N_PARALLEL_ENVS = 1
STEPS_PER_CYCLE = 128


# Define the linear epsilon function
def linear_epsilon(steps_done):
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * (steps_done / EPS_DECAY))

# Define the corrected exponential epsilon function
def exponential_epsilon(steps_done):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)

env = Environment(map_file=MAP_FILE,
                  n_robots=NUM_AGENTS, 
                  n_packages=N_PACKAGES,
                  move_cost=MOVE_COST,
                  delivery_reward=DELIVERY_REWARD,
                  delay_reward=DELAY_REWARD,
                  seed=SEED,
                  max_time_steps=MAX_TIME_STEPS)

env.reset()
    
vec_env = VectorizedEnv(
    Environment, num_envs=N_PARALLEL_ENVS,
    map_file=MAP_FILE,
    n_robots=NUM_AGENTS,
    n_packages=N_PACKAGES,
    move_cost=MOVE_COST,
    delivery_reward=DELIVERY_REWARD,
    delay_reward=DELAY_REWARD,
    seed=SEED,
    max_time_steps=MAX_TIME_STEPS
)

class QMixTrainer:
    def __init__(self, vec_env, lr=LR, weight_decay=WEIGHT_DECAY,
                 gamma=GAMMA, tau=TAU,
                 gradient_clipping=GRADIENT_CLIPPING,
                 use_data_parallel=True, num_envs=1):
        self.env = vec_env
        self.num_envs = num_envs
        ref_env = self.env.envs[0]
        self.OBS_DIM = (6, ref_env.n_rows, ref_env.n_cols)
        self.STATE_DIM = (7, ref_env.n_rows, ref_env.n_cols)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_data_parallel = use_data_parallel

        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau
        self.gradient_clipping = gradient_clipping
        
        self.persistent_packages_list = [{} for _ in range(self.num_envs)]
        
        self.le_move = LabelEncoder()
        self.le_move.fit(['S', 'L', 'R', 'U', 'D'])
        self.le_pkg_op = LabelEncoder()
        self.le_pkg_op.fit(['0', '1', '2'])
        self.NUM_MOVE_ACTIONS = len(self.le_move.classes_)
        self.NUM_PKG_OPS = len(self.le_pkg_op.classes_)
        
        self.agent_network = AgentNetwork(self.OBS_DIM, ACTION_DIM).to(self.device)
        self.mixing_network = MixingNetwork(self.STATE_DIM, NUM_AGENTS, MIXING_DIM).to(self.device)
        
        self.target_agent_network = AgentNetwork(self.OBS_DIM, ACTION_DIM).to(self.device)
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixing_network = MixingNetwork(self.STATE_DIM, NUM_AGENTS, MIXING_DIM).to(self.device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        self.agent_network = nn.DataParallel(self.agent_network)
        self.mixing_network = nn.DataParallel(self.mixing_network)
        self.target_agent_network = nn.DataParallel(self.target_agent_network)
        self.target_mixing_network = nn.DataParallel(self.target_mixing_network)

        if self.use_data_parallel:
            self.target_agent_network.module.load_state_dict(self.agent_network.module.state_dict())
            self.target_mixing_network.module.load_state_dict(self.mixing_network.module.state_dict())
        else:
            self.target_agent_network.load_state_dict(self.agent_network.state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        if self.use_data_parallel:
            self.agent_optimizer = optim.RMSprop(self.agent_network.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.mixing_optimizer = optim.RMSprop(self.mixing_network.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.agent_optimizer = optim.RMSprop(self.agent_network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.mixing_optimizer = optim.RMSprop(self.mixing_network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.buffer = ReplayBuffer(capacity=MAX_REPLAY_BUFFER_SIZE, 
                                   num_agents=NUM_AGENTS, 
                                   obs_shape=self.OBS_DIM, 
                                   state_shape=self.STATE_DIM, 
                                   device=self.device)

        self.update_targets(1.0)

        self.current_env_state_dicts = [env.reset() for env in self.env.envs]
        for i in range(self.num_envs):
            self._update_persistent_packages(self.current_env_state_dicts[i], i)
        self.env_dones = [False] * self.num_envs
        self.env_episode_rewards = [0.0] * self.num_envs
        self.env_episode_steps = [0] * self.num_envs

        # --- Add: Hidden state trackers for each agent in each env ---
        self.agent_hidden_states = [
            [None for _ in range(NUM_AGENTS)] for _ in range(self.num_envs)
        ]
        
    def _update_persistent_packages(self, current_env_state, env_idx):
        pkgs_for_env = self.persistent_packages_list[env_idx]

        if 'packages' in current_env_state and current_env_state['packages'] is not None:
            for pkg_tuple in current_env_state['packages']:
                pkg_id = pkg_tuple[0]
                if pkg_id not in pkgs_for_env:
                    pkgs_for_env[pkg_id] = {
                        'id': pkg_id,
                        'start_pos': (pkg_tuple[1] - 1, pkg_tuple[2] - 1),
                        'target_pos': (pkg_tuple[3] - 1, pkg_tuple[4] - 1),
                        'start_time': pkg_tuple[5],
                        'deadline': pkg_tuple[6],
                        'status': 'waiting'
                    }

        current_carried_pkg_ids_set = set()
        if 'robots' in current_env_state and current_env_state['robots'] is not None:
            for r_idx, r_data in enumerate(current_env_state['robots']):
                carried_id = r_data[2]
                if carried_id != 0:
                    current_carried_pkg_ids_set.add(carried_id)

        packages_to_remove_definitively = []

        for pkg_id, pkg_data in list(pkgs_for_env.items()):
            original_status_in_tracker = pkg_data['status']

            if pkg_id in current_carried_pkg_ids_set:
                pkgs_for_env[pkg_id]['status'] = 'in_transit'
            else:
                if original_status_in_tracker == 'in_transit':
                    packages_to_remove_definitively.append(pkg_id)
                pass

        for pkg_id_to_remove in packages_to_remove_definitively:
            if pkg_id_to_remove in pkgs_for_env:
                del pkgs_for_env[pkg_id_to_remove]


    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau
        agent_params_src = self.agent_network.module.parameters() if self.use_data_parallel else self.agent_network.parameters()
        agent_params_tgt = self.target_agent_network.module.parameters() if self.use_data_parallel else self.target_agent_network.parameters()
        mixer_params_src = self.mixing_network.module.parameters() if self.use_data_parallel else self.mixing_network.parameters()
        mixer_params_tgt = self.target_mixing_network.module.parameters() if self.use_data_parallel else self.target_mixing_network.parameters()

        for target_param, param in zip(agent_params_tgt, agent_params_src):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(mixer_params_tgt, mixer_params_src):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self, path_prefix="qmix"):
        dir_name = os.path.dirname(path_prefix)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        agent_state_dict = self.agent_network.module.state_dict() if self.use_data_parallel else self.agent_network.state_dict()
        mixer_state_dict = self.mixing_network.module.state_dict() if self.use_data_parallel else self.mixing_network.state_dict()

        torch.save(agent_state_dict, f"{path_prefix}_agent.pt")
        torch.save(mixer_state_dict, f"{path_prefix}_mixer.pt")
        print(f"Models saved with prefix {path_prefix}")
        
    def load_models(self, path_prefix="qmix"):
        agent_path = f"{path_prefix}_agent.pt"
        mixer_path = f"{path_prefix}_mixer.pt"

        if self.use_data_parallel:
            self.agent_network.module.load_state_dict(torch.load(agent_path, map_location=self.device))
            self.mixing_network.module.load_state_dict(torch.load(mixer_path, map_location=self.device))
        else:
            self.agent_network.load_state_dict(torch.load(agent_path, map_location=self.device))
            self.mixing_network.load_state_dict(torch.load(mixer_path, map_location=self.device))
        self.update_targets(tau=1.0)
        print(f"Models loaded from prefix {path_prefix}")
    
    def select_action(self, obs, eps, agent_idx=None, env_idx=None):
        if not isinstance(obs, torch.Tensor) and isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        # Add a batch dimension: (C, H, W) -> (1, C, H, W)
        obs_for_network = obs.unsqueeze(0)

        # --- Add: Use hidden state if provided ---
        hidden = None
        if agent_idx is not None and env_idx is not None:
            hidden = self.agent_hidden_states[env_idx][agent_idx]
            if hidden is not None and not isinstance(hidden, torch.Tensor):
                hidden = hidden.to(self.device)

        if np.random.rand() < eps:
            action = np.random.randint(0, ACTION_DIM)
            # --- Add: Reset hidden state if random action (optional, or keep as is) ---
        else:
            with torch.no_grad():
                q_values, new_hidden = self.agent_network(obs_for_network, hidden_state=hidden)
                action = torch.argmax(q_values.squeeze(0), dim=0).item()
                # --- Add: Update hidden state ---
                if agent_idx is not None and env_idx is not None:
                    self.agent_hidden_states[env_idx][agent_idx] = new_hidden
        return action

    def train_step(self, batch_size):
        if not self.buffer.can_sample(batch_size):
            return None

        batch_obs, batch_next_obs, batch_actions, reward_tot, dones, batch_states, batch_next_states = \
            self.buffer.sample(batch_size)
        
        # batch_obs shape: (B, num_agents, C, H, W)
        # batch_actions shape: (B, num_agents)

        # --- Current Q-values ---
        # self.agent_network now returns (q_values, hidden_state). We need only q_values.
        current_q_s_list = []
        for i in range(NUM_AGENTS):
            obs_agent_i = batch_obs[:, i] # Shape: (B, C, H, W)
            # ---- START DEBUG PRINT ----
            if i == 0: # Print only for the first agent to avoid too much output
                print(f"Debug: obs_agent_i shape: {obs_agent_i.shape}")
            # ---- END DEBUG PRINT ----
            q_values_agent_i, _ = self.agent_network(obs_agent_i) # Extract Q-values
            current_q_s_list.append(q_values_agent_i)
        
        current_q_s = torch.stack(current_q_s_list, dim=1) # Shape: (B, num_agents, ACTION_DIM)
        
        chosen_action_qvals = torch.gather(current_q_s, dim=2, index=batch_actions.unsqueeze(2)).squeeze(2) # Shape: (B, num_agents)
        q_tot = self.mixing_network(chosen_action_qvals, batch_states) # Shape: (B,)

        # --- Target Q-values ---
        # self.target_agent_network also returns (q_values, hidden_state).
        target_q_s_list = []
        with torch.no_grad():
            for i in range(NUM_AGENTS):
                next_obs_agent_i = batch_next_obs[:, i] # Shape: (B, C, H, W)
                # Target agent network processes batch, returns (q_values_batch, hidden_state_batch)
                target_q_values_agent_i, _ = self.target_agent_network(next_obs_agent_i) # Extract Q-values
                target_q_s_list.append(target_q_values_agent_i)
            
            target_q_values = torch.stack(target_q_s_list, dim=1) # Shape: (B, num_agents, ACTION_DIM)

            target_max_q_values = target_q_values.max(dim=2)[0] # Shape: (B, num_agents)
            next_q_tot = self.target_mixing_network(target_max_q_values, batch_next_states) # Shape: (B,)

        target_q_tot = reward_tot + self.gamma * (1 - dones) * next_q_tot
        
        loss = F.mse_loss(q_tot, target_q_tot.detach())

        self.agent_optimizer.zero_grad()
        self.mixing_optimizer.zero_grad()
        loss.backward()
        agent_params_to_clip = self.agent_network.module.parameters() if self.use_data_parallel else self.agent_network.parameters()
        mixer_params_to_clip = self.mixing_network.module.parameters() if self.use_data_parallel else self.mixing_network.parameters()
        torch.nn.utils.clip_grad_norm_(agent_params_to_clip, self.gradient_clipping)
        torch.nn.utils.clip_grad_norm_(mixer_params_to_clip, self.gradient_clipping)
        self.agent_optimizer.step()
        self.mixing_optimizer.step()

        self.update_targets(self.tau)

        return loss.item()


    def collect_and_train_cycle(self, eps, num_env_steps_this_cycle):
        cycle_completed_episode_rewards = []
        cycle_total_loss = 0.0
        cycle_train_steps = 0

        for _ in range(num_env_steps_this_cycle):
            for env_idx in range(self.num_envs):
                if self.env_dones[env_idx]:
                    cycle_completed_episode_rewards.append(self.env_episode_rewards[env_idx])
                    
                    self.current_env_state_dicts[env_idx] = self.env.envs[env_idx].reset()
                    self.persistent_packages_list[env_idx] = {}
                    self._update_persistent_packages(self.current_env_state_dicts[env_idx], env_idx)
                    self.env_dones[env_idx] = False
                    self.env_episode_rewards[env_idx] = 0.0
                    self.env_episode_steps[env_idx] = 0
                    # --- Add: Reset hidden states for this env ---
                    self.agent_hidden_states[env_idx] = [None for _ in range(NUM_AGENTS)]

                current_s_dict = self.current_env_state_dicts[env_idx]
                
                observations_this_env = []
                actions_this_env_int = []
                env_actions_this_env_tuple = []

                for agent_i in range(NUM_AGENTS):
                    obs = convert_state(current_s_dict, self.persistent_packages_list[env_idx], current_robot_idx=agent_i)
                    observations_this_env.append(obs)
                    # --- Pass agent_idx and env_idx to select_action ---
                    action_int = self.select_action(obs, eps, agent_idx=agent_i, env_idx=env_idx)
                    actions_this_env_int.append(action_int)

                    move_idx = action_int % self.NUM_MOVE_ACTIONS
                    pkg_op_idx = action_int // self.NUM_MOVE_ACTIONS
                    if pkg_op_idx >= self.NUM_PKG_OPS:
                        pkg_op_idx = 0 
                    move_str = self.le_move.inverse_transform([move_idx])[0]
                    pkg_op_str = self.le_pkg_op.inverse_transform([pkg_op_idx])[0]
                    env_actions_this_env_tuple.append((move_str, pkg_op_str))
                
                prev_state_tensor = convert_global_state_to_tensor(current_s_dict, 
                                                                   self.persistent_packages_list[env_idx],
                                                                   self.STATE_DIM)
                
                next_s_dict, global_reward, done, _ = self.env.envs[env_idx].step(env_actions_this_env_tuple)
                self._update_persistent_packages(next_s_dict, env_idx)
                
                current_state_tensor = convert_global_state_to_tensor(next_s_dict, 
                                                                    self.persistent_packages_list[env_idx],
                                                                    self.STATE_DIM)
                
                next_observations_this_env = []
                for agent_i in range(NUM_AGENTS):
                    next_obs = convert_state(next_s_dict, self.persistent_packages_list[env_idx], current_robot_idx=agent_i)
                    next_observations_this_env.append(next_obs)

                self.buffer.add(
                    obs=observations_this_env,
                    next_obs=next_observations_this_env,
                    state=prev_state_tensor,
                    next_state=current_state_tensor,
                    actions=actions_this_env_int,
                    total_reward=global_reward,
                    done=done
                )

                self.env_episode_rewards[env_idx] += global_reward
                self.env_episode_steps[env_idx] += 1
                self.current_env_state_dicts[env_idx] = next_s_dict
                self.env_dones[env_idx] = done
            
            loss = self.train_step(BATCH_SIZE)
            if loss is not None:
                cycle_total_loss += loss
                cycle_train_steps += 1

        avg_reward_this_cycle = sum(cycle_completed_episode_rewards) / max(1, len(cycle_completed_episode_rewards)) if cycle_completed_episode_rewards else 0.0
        avg_loss_this_cycle = cycle_total_loss / max(1, cycle_train_steps) if cycle_train_steps > 0 else 0.0
        
        return avg_reward_this_cycle, avg_loss_this_cycle

    def run_episode(self, eps):
        # Vectorized reset
        current_state_dicts = self.env.reset()
        persistent_packages_list = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            self.persistent_packages = persistent_packages_list[i]
            self._update_persistent_packages(current_state_dicts[i])

        dones = [False] * self.num_envs
        episode_rewards = [0.0] * self.num_envs
        episode_losses = [0.0] * self.num_envs
        step_count = 0

        while not all(dones):
            actions_batch = []
            observations_batch = []
            env_actions_batch = []
            packages_before_action_batch = []

            for env_idx in range(self.num_envs):
                if dones[env_idx]:
                    actions_batch.append([0]*NUM_AGENTS)
                    observations_batch.append([np.zeros((6, self.env.envs[env_idx].n_rows, self.env.envs[env_idx].n_cols)) for _ in range(NUM_AGENTS)])
                    env_actions_batch.append([('S', '0')]*NUM_AGENTS)
                    packages_before_action_batch.append({})
                    continue

                self.persistent_packages = persistent_packages_list[env_idx]
                obs_list = []
                act_list = []
                for i in range(NUM_AGENTS):
                    obs = convert_state(current_state_dicts[env_idx], self.persistent_packages, current_robot_idx=i)
                    obs_list.append(obs)
                    act = self.select_action(obs, eps)
                    act_list.append(act)
                observations_batch.append(obs_list)
                actions_batch.append(act_list)
                env_actions = []
                for int_act in act_list:
                    move_idx = int_act % self.NUM_MOVE_ACTIONS
                    pkg_op_idx = int_act // self.NUM_MOVE_ACTIONS
                    if pkg_op_idx >= self.NUM_PKG_OPS:
                        pkg_op_idx = 0
                    move_str = self.le_move.inverse_transform([move_idx])[0]
                    pkg_op_str = self.le_pkg_op.inverse_transform([pkg_op_idx])[0]
                    env_actions.append((move_str, pkg_op_str))
                env_actions_batch.append(env_actions)
                packages_before_action_batch.append(copy.deepcopy(self.persistent_packages))

            # Step all envs at once
            next_state_dicts, global_rewards, step_dones, _ = self.env.step(env_actions_batch)
            for env_idx in range(self.num_envs):
                if dones[env_idx]:
                    continue
                self.persistent_packages = persistent_packages_list[env_idx]
                # ... rest of your per-env logic ...


if __name__ == "__main__":

    trainer = QMixTrainer(vec_env=vec_env, use_data_parallel=True, num_envs=N_PARALLEL_ENVS)

    episode_rewards_history = []
    episode_avg_loss_history = []

    training_completed_successfully = False
    print("Starting QMIX training with parallel environments...")
    print(f"Running for {NUM_EPISODES} training cycles.")
    print(f"Number of parallel environments: {N_PARALLEL_ENVS}, Steps per cycle per environment: {STEPS_PER_CYCLE}")

    try:
        for cycle_num in range(1, NUM_EPISODES + 1):
            current_epsilon = linear_epsilon(cycle_num - 1) 
            
            avg_reward, avg_loss = trainer.collect_and_train_cycle(current_epsilon, STEPS_PER_CYCLE)
            
            episode_rewards_history.append(avg_reward)
            episode_avg_loss_history.append(avg_loss)
            
            if cycle_num % 10 == 0 or cycle_num == NUM_EPISODES: 
                print(
                    f"[Cycle {cycle_num}/{NUM_EPISODES}] "
                    f"Avg Reward (per completed episode across all envs): {avg_reward:.2f} | "
                    f"Avg Training Loss: {avg_loss:.4f} | "
                    f"Epsilon: {current_epsilon:.3f}"
                )

            if cycle_num % 50 == 0: 
                print(f"Saving checkpoint at cycle {cycle_num}...")
                trainer.save_models(path_prefix=f"models/qmix_agent_cycle{cycle_num}")
                trainer.save_models(path_prefix=f"models/qmix_mixer_cycle{cycle_num}")
                
        training_completed_successfully = True

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt).")
        print("Saving current model state...")
        trainer.save_models(path_prefix="models/qmix_interrupted")
        print("Models saved to _interrupted.pt files.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("Saving model due to exception...")
        trainer.save_models(path_prefix="models/qmix_exception")
        print("Models saved to _exception.pt files.")
    finally:
        print("\nTraining loop finished or was interrupted.")
        
        if episode_rewards_history:
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards_history, label='Raw reward')
            
            # --- Moving average ---
            window = 20
            if len(episode_rewards_history) >= window:
                rewards = np.array(episode_rewards_history)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(np.arange(window-1, len(rewards)), moving_avg, label=f'Moving avg ({window})')
            
            plt.title('Average Total Reward per Cycle (across all envs)')
            plt.xlabel('Cycle')
            plt.ylabel('Average Total Reward')
            plt.grid(True)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(episode_avg_loss_history)
            plt.title('Average Loss per Cycle')
            plt.xlabel('Cycle')
            plt.ylabel('Average Loss')
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("No data recorded for plotting.")

    if training_completed_successfully:
        print("\nTraining completed successfully.")
        print("Saving final model...")
        trainer.save_models(path_prefix="models/qmix_final")
        print("Final models saved.")