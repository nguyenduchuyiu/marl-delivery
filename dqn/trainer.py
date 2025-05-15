#trainer.py
import pygame
from networks import AgentNetwork, ReplayBuffer, convert_state, reward_shaping
from env import Environment
from env_vectorized import VectorizedEnv

import torch
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
MOVE_COST = -0.1
DELIVERY_REWARD = 0
DELAY_REWARD = 0
MAX_TIME_STEPS = 1000
NUM_EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-5
WEIGHT_DECAY = 1e-4
MAX_REPLAY_BUFFER_SIZE = 10000
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.01
GRADIENT_CLIPPING = 10

NUM_ENVS = 4  # You can adjust this for your hardware

# Replace single env with vectorized env
vec_env = VectorizedEnv(
    Environment, num_envs=NUM_ENVS,
    map_file=MAP_FILE,
    n_robots=NUM_AGENTS,
    n_packages=N_PACKAGES,
    move_cost=MOVE_COST,
    delivery_reward=DELIVERY_REWARD,
    delay_reward=DELAY_REWARD,
    seed=SEED,
    max_time_steps=MAX_TIME_STEPS
)

# Define the linear epsilon function
def linear_epsilon(steps_done):
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * (steps_done / EPS_DECAY))

# Define the corrected exponential epsilon function
def exponential_epsilon(steps_done):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)

    
def save_model(policy_net, path="models/dqn_agent.pt"):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(policy_net.state_dict(), path)
    
env = Environment(map_file=MAP_FILE,
                  n_robots=NUM_AGENTS, 
                  n_packages=N_PACKAGES,
                  move_cost=MOVE_COST,
                  delivery_reward=DELIVERY_REWARD,
                  delay_reward=DELAY_REWARD,
                  seed=SEED,
                  max_time_steps=MAX_TIME_STEPS)

env.reset()



class DQNTrainer:
    def __init__(self, env, lr=LR, weight_decay=WEIGHT_DECAY, gamma=GAMMA, tau=TAU, gradient_clipping=GRADIENT_CLIPPING, num_envs=1):
        self.env = env
        self.num_envs = num_envs
        OBS_DIM = (6, env.envs[0].n_rows, env.envs[0].n_cols) if hasattr(env, 'envs') else (6, env.n_rows, env.n_cols)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize LabelEncoders for actions
        self.le_move = LabelEncoder()
        self.le_move.fit(['S', 'L', 'R', 'U', 'D']) # Stay, Left, Right, Up, Down
        self.le_pkg_op = LabelEncoder()
        self.le_pkg_op.fit(['0', '1', '2']) # 0: None, 1: Pickup, 2: Drop
        self.NUM_MOVE_ACTIONS = len(self.le_move.classes_) # 5
        self.NUM_PKG_OPS = len(self.le_pkg_op.classes_) # 3
        
        # Network
        self.agent_network = AgentNetwork(OBS_DIM, ACTION_DIM).to(self.device)
        
        # Target networks
        self.target_agent_network = AgentNetwork(OBS_DIM, ACTION_DIM).to(self.device)
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        
        self.buffer = ReplayBuffer(capacity=MAX_REPLAY_BUFFER_SIZE, obs_shape=OBS_DIM, device=self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau
        self.gradient_clipping = gradient_clipping
        
        # Persistent packages to track the package id and the target position
        self.persistent_packages = {}

        self.update_targets(1.0)  # Hard update at start

        self.agent_optimizer = optim.Adam(self.agent_network.parameters(), lr=lr, weight_decay=weight_decay)
        
    def _update_persistent_packages(self, current_env_state): 
        """
        Updates self.persistent_packages based on the current environment state.
        - current_env_state: The state dictionary from env.step() or env.reset().
        """
        # 1. Add newly appeared packages to persistent_packages if not already tracked
        if 'packages' in current_env_state and current_env_state['packages'] is not None:
            for pkg_tuple in current_env_state['packages']:
                pkg_id = pkg_tuple[0]
                if pkg_id not in self.persistent_packages:
                    self.persistent_packages[pkg_id] = {
                        'id': pkg_id,
                        'start_pos': (pkg_tuple[1] - 1, pkg_tuple[2] - 1),
                        'target_pos': (pkg_tuple[3] - 1, pkg_tuple[4] - 1),
                        'start_time': pkg_tuple[5],
                        'deadline': pkg_tuple[6],
                        'status': 'waiting'
                    }

        # 2. Get current robot carrying info
        current_carried_pkg_ids_set = set()
        if 'robots' in current_env_state and current_env_state['robots'] is not None:
            for r_idx, r_data in enumerate(current_env_state['robots']):
                carried_id = r_data[2] # (pos_x+1, pos_y+1, carrying_package_id)
                if carried_id != 0:
                    current_carried_pkg_ids_set.add(carried_id)

        packages_to_remove_definitively = []

        # 3. Update package status
        for pkg_id, pkg_data in list(self.persistent_packages.items()):
            original_status_in_tracker = pkg_data['status']

            if pkg_id in current_carried_pkg_ids_set:
                # If currently being carried by any robot in current_env_state, set to 'in_transit'
                self.persistent_packages[pkg_id]['status'] = 'in_transit'
            else:
                # Package is NOT being carried in current_env_state
                if original_status_in_tracker == 'in_transit':
                    # This package WAS 'in_transit' (according to our tracker)
                    # and is now NOT carried in current_env_state.
                    # Given the env.py logic, this means it MUST have been delivered correctly.
                    packages_to_remove_definitively.append(pkg_id)
                # If original_status_in_tracker was 'waiting' and it's still not carried,
                # its status remains 'waiting'. No change needed to start_pos or status here.
                pass

        # 4. Remove packages that were successfully delivered
        for pkg_id_to_remove in packages_to_remove_definitively:
            if pkg_id_to_remove in self.persistent_packages:
                del self.persistent_packages[pkg_id_to_remove]


    def update_targets(self, tau=None):
        if tau is None:
            tau = self.tau
        # Soft update
        for target_param, param in zip(self.target_agent_network.parameters(), self.agent_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    def select_action(self, obs, eps):
        # obs_batch: (C, H, W)
        if obs is not isinstance(obs, torch.Tensor) and isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        if np.random.rand() < eps:
            action = np.random.randint(0, ACTION_DIM)
        else:
            with torch.no_grad():
                q_values = self.agent_network(obs)
                action = torch.argmax(q_values, dim=1).item()
        return action

    def train_step(self, batch_size):
        if not self.buffer.can_sample(batch_size):
            return None

        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self.buffer.sample(batch_size)
        # batch_obs: (B, C, H, W)
        # batch_actions: (B,)
        # batch_rewards: (B,)
        # batch_next_obs: (B, C, H, W)
        # batch_dones: (B,)

        # Compute Q(s, a) for the actions taken
        q_values = self.agent_network(batch_obs)  # (B, action_dim)
        q_value = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_agent_network(batch_next_obs)  # (B, action_dim)
            next_q_value = next_q_values.max(1)[0]  # (B,)
            target = batch_rewards + self.gamma * (1 - batch_dones) * next_q_value

        # Loss
        loss = F.mse_loss(q_value, target)

        # Optimize
        self.agent_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), self.gradient_clipping)
        self.agent_optimizer.step()

        # Soft update target network
        self.update_targets(self.tau)

        return loss.item()


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
            # Comment out rendering for speed
            # self.env.render()
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

            # Step all envs
            next_state_dicts, global_rewards, step_dones, _ = self.env.step(env_actions_batch)
            for env_idx in range(self.num_envs):
                if dones[env_idx]:
                    continue
                self.persistent_packages = persistent_packages_list[env_idx]
                individual_rewards = reward_shaping(
                    current_state_dicts[env_idx],
                    next_state_dicts[env_idx],
                    env_actions_batch[env_idx],
                    packages_before_action_batch[env_idx],
                    NUM_AGENTS
                )
                self._update_persistent_packages(next_state_dicts[env_idx])
                # Build per-agent next observations
                next_obs_list = []
                for i in range(NUM_AGENTS):
                    next_obs = convert_state(next_state_dicts[env_idx], self.persistent_packages, current_robot_idx=i)
                    next_obs_list.append(next_obs)
                # Store in buffer
                for i in range(NUM_AGENTS):
                    self.buffer.add(
                        obs=observations_batch[env_idx][i],
                        action=actions_batch[env_idx][i],
                        reward=individual_rewards[i],
                        next_obs=next_obs_list[i],
                        done=step_dones[env_idx]
                    )
                episode_rewards[env_idx] += global_rewards[env_idx]
            current_state_dicts = next_state_dicts
            dones = [d or sd for d, sd in zip(dones, step_dones)]
            step_count += 1
            # Training step
            loss = self.train_step(BATCH_SIZE)
            if loss is not None:
                for env_idx in range(self.num_envs):
                    episode_losses[env_idx] += loss

        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean(episode_losses) / max(1, step_count)
        return avg_reward, avg_loss



if __name__ == "__main__":
    
    import pygame
    trainer = DQNTrainer(vec_env, num_envs=NUM_ENVS)

    # Lists to store metrics for plotting
    episode_rewards_history = []
    episode_avg_loss_history = []

    training_completed_successfully = False
    print("Starting DQN training...")
    print(f"Running for {NUM_EPISODES} episodes.")

    try:
        for episode_num in range(1, NUM_EPISODES + 1):
            # The exponential_epsilon function from in[16] expects 'steps_done'
            # Assuming 'steps_done' in that context refers to the number of episodes completed (0-indexed)
            current_epsilon = linear_epsilon(episode_num - 1) 
            
            episode_reward, avg_episode_loss = trainer.run_episode(current_epsilon)
            
            episode_rewards_history.append(episode_reward)
            episode_avg_loss_history.append(avg_episode_loss)
            
            if episode_num % 10 == 0 or episode_num == NUM_EPISODES: # Print every 10 episodes and the last one
                print(f"Episode {episode_num}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Avg Loss: {avg_episode_loss:.4f} | Epsilon: {current_epsilon:.3f}")

            # Optional: Periodic saving during training
            if episode_num % 50 == 0: # Example: Save every 50 episodes
                print(f"Saving checkpoint at episode {episode_num}...")
                save_model(trainer.agent_network, path=f"models/dqn_agent_ep{episode_num}.pt")
                
        training_completed_successfully = True

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt).")
        print("Saving current model state...")
        save_model(trainer.agent_network, path="models/dqn_agent_interrupted.pt")
        print("Models saved to _interrupted.pt files.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for the exception
        print("Saving model due to exception...")
        save_model(trainer.agent_network, path="models/dqn_agent_exception.pt")
        print("Models saved to _exception.pt files.")
    finally:
        pygame.quit()
        print("\nTraining loop finished or was interrupted.")
        
        # Plotting the results
        if episode_rewards_history: # Check if there's any data to plot
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards_history)
            plt.title('Total Reward per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(episode_avg_loss_history)
            plt.title('Average Loss per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Average Loss')  
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("No data recorded for plotting.")

    if training_completed_successfully:
        print("\nTraining completed successfully.")
        print("Saving final model...")
        save_model(trainer.agent_network, path="models/dqn_agent_final.pt")
        print("Final models saved.")