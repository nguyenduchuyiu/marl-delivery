import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder # Đã sửa thành sklearn.preprocessing
from env import Environment
from networks import ReplayBuffer, convert_state

# --- Định nghĩa lại các class và hàm cần thiết (Lấy từ code của bạn) ---
# (Environment, AgentNetwork, MixingNetwork, ReplayBuffer, convert_state)
# ... (Dán code của các class này vào đây) ...
# Ví dụ:
# class Environment: ...
# class AgentNetwork(nn.Module): ...
# class HyperNetwork(nn.Module): ...
# class MixingNetwork(nn.Module): ...
# class ReplayBuffer: ...
# def convert_state(state, persistent_packages, current_robot_idx=None): ...
# (Đảm bảo convert_state là phiên bản đã sửa với 8/6 channel và scaling)

# --- Các hằng số cần thiết ---
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

ACTION_DIM = 15
NUM_AGENTS = 5
MAP_FILE = "map2.txt" # Hoặc map.txt tùy bạn
N_PACKAGES = 6 # Giảm số lượng gói hàng để dễ theo dõi hơn trong test
MOVE_COST = -0.01
DELIVERY_REWARD = 10
DELAY_REWARD = 1
MAX_TIME_STEPS = 20 # Giảm max time steps cho test
MIXING_DIM = 64 # Giảm cho test
# OBS_DIM và STATE_DIM sẽ được xác định dựa trên output của convert_state
# (Giả sử convert_state trả về 8 channel cho obs, 6 cho state)
# Ví dụ:
# OBS_SHAPE_FROM_CONVERT = (8, 20, 20) # (C, H, W) - Cần khớp với map_file
# STATE_SHAPE_FROM_CONVERT = (6, 20, 20) # (C, H, W) - Cần khớp với map_file

MAX_REPLAY_BUFFER_SIZE = 1000 # Buffer nhỏ cho test
BATCH_SIZE_FOR_TEST = 4 # Số lượng mẫu lấy ra từ buffer để test

# --- Helper function để lấy persistent_packages (giống code của bạn) ---
def get_persistent_packages_from_env(env_instance):
    # Cần truy cập trực tiếp vào self.packages của instance env
    # hoặc env_instance phải có một phương thức để lấy chúng.
    # Giả sử env_instance.packages là một list các đối tượng Package
    persistent = {}
    for p_obj in env_instance.packages: # Giả sử env_instance.packages là list các Package objects
        # Chỉ thêm những gói hàng chưa được giao hoặc vẫn còn "active"
        # Điều kiện này có thể cần điều chỉnh tùy theo logic của bạn
        if p_obj.status != 'delivered' and p_obj.status != 'None': # 'None' là trạng thái ban đầu trước khi xuất hiện
            persistent[p_obj.package_id] = {
                'id': p_obj.package_id,
                'start_pos': p_obj.start,
                'target_pos': p_obj.target,
                'start_time': p_obj.start_time,
                'deadline': p_obj.deadline,
                'status': p_obj.status # Quan trọng: status này phải được cập nhật đúng
            }
    return persistent

# --- Khởi tạo môi trường ---
test_env = Environment(map_file=MAP_FILE,
                       n_robots=NUM_AGENTS,
                       n_packages=N_PACKAGES,
                       max_time_steps=MAX_TIME_STEPS,
                       seed=SEED)

# Xác định OBS_DIM và STATE_DIM từ một mẫu convert_state
_initial_state_dict_for_shape = test_env.reset()
_initial_persistent_pkgs_for_shape = get_persistent_packages_from_env(test_env)
_sample_obs_np = convert_state(_initial_state_dict_for_shape, _initial_persistent_pkgs_for_shape, current_robot_idx=0)
_sample_state_np = convert_state(_initial_state_dict_for_shape, _initial_persistent_pkgs_for_shape, current_robot_idx=None)

OBS_SHAPE_FROM_CONVERT = _sample_obs_np.shape # (C, H, W)
STATE_SHAPE_FROM_CONVERT = _sample_state_np.shape # (C, H, W)

print(f"Detected OBS_SHAPE: {OBS_SHAPE_FROM_CONVERT}")
print(f"Detected STATE_SHAPE: {STATE_SHAPE_FROM_CONVERT}")

# --- Khởi tạo Replay Buffer ---
replay_buffer = ReplayBuffer(capacity=MAX_REPLAY_BUFFER_SIZE,
                             num_agents=NUM_AGENTS,
                             obs_shape=OBS_SHAPE_FROM_CONVERT,
                             state_shape=STATE_SHAPE_FROM_CONVERT,
                             device=device)

# --- Mô phỏng việc chạy một vài episode và thêm vào buffer ---
num_episodes_to_simulate = 3
persistent_packages_current = {} # Sẽ được cập nhật trong vòng lặp

for ep in range(num_episodes_to_simulate):
    current_state_dict = test_env.reset()
    # Cập nhật persistent_packages từ trạng thái reset của env
    # (Giả sử hàm _update_persistent_packages của QMixTrainer làm điều này)
    # Ở đây, ta sẽ gọi get_persistent_packages_from_env để mô phỏng
    persistent_packages_current = get_persistent_packages_from_env(test_env)
    # In ra các gói hàng đang được theo dõi để debug
    # print(f"Episode {ep+1}, Initial tracked packages: {list(persistent_packages_current.keys())}")


    for t in range(MAX_TIME_STEPS):
        # Tạo observation và global state tại thời điểm t
        observations_t_np = []
        actions_t_int = []
        env_actions_t_str = []

        # Copy persistent_packages cho bước hiện tại để đảm bảo tính nhất quán
        persistent_packages_for_current_step = persistent_packages_current.copy()

        for i in range(NUM_AGENTS):
            obs_np_i_t = convert_state(current_state_dict, persistent_packages_for_current_step, current_robot_idx=i)
            observations_t_np.append(obs_np_i_t)
            # Hành động ngẫu nhiên cho test
            action_int_i_t = random.randint(0, ACTION_DIM - 1)
            actions_t_int.append(action_int_i_t)

            # Decode action (không thực sự cần LabelEncoder ở đây cho test ngẫu nhiên)
            # move_idx = action_int_i_t % 5 # Giả sử 5 hành động di chuyển
            # pkg_op_idx = action_int_i_t // 5
            # env_actions_t_str.append((str(move_idx), str(pkg_op_idx))) # Đơn giản hóa
            # Sử dụng hành động đơn giản: tất cả agent đứng yên, không làm gì
            env_actions_t_str.append(('S', '0'))


        global_state_t_np = convert_state(current_state_dict, persistent_packages_for_current_step, current_robot_idx=None)

        # Thực hiện hành động
        next_env_state_dict, global_reward, done, _ = test_env.step(env_actions_t_str)

        # Cập nhật persistent_packages dựa trên trạng thái mới
        # Trong QMixTrainer, đây sẽ là self._update_persistent_packages(next_env_state_dict)
        # Ở đây, ta mô phỏng lại:
        persistent_packages_current = get_persistent_packages_from_env(test_env)
        # print(f"  Step {t+1}, Tracked packages after update: {list(persistent_packages_current.keys())}")


        # Tạo observation và global state tại thời điểm t+1
        next_observations_t_plus_1_np = []
        persistent_packages_for_next_step = persistent_packages_current.copy()

        for i in range(NUM_AGENTS):
            next_obs_np_i_t_plus_1 = convert_state(next_env_state_dict, persistent_packages_for_next_step, current_robot_idx=i)
            next_observations_t_plus_1_np.append(next_obs_np_i_t_plus_1)

        next_global_state_t_plus_1_np = convert_state(next_env_state_dict, persistent_packages_for_next_step, current_robot_idx=None)

        # Thêm vào buffer
        replay_buffer.add(
            obs=np.array(observations_t_np),
            next_obs=np.array(next_observations_t_plus_1_np),
            state=global_state_t_np,
            next_state=next_global_state_t_plus_1_np,
            actions=np.array(actions_t_int),
            total_reward=global_reward,
            done=done
        )

        current_state_dict = next_env_state_dict
        if done:
            break
    print(f"Episode {ep+1} finished after {t+1} steps. Buffer size: {len(replay_buffer)}")


# --- Lấy mẫu từ Replay Buffer và Visualize ---
if replay_buffer.can_sample(BATCH_SIZE_FOR_TEST):
    print(f"\nSampling {BATCH_SIZE_FOR_TEST} transitions from buffer...")
    batch_states, batch_next_states, batch_obs, batch_next_obs, \
    batch_actions, batch_total_rewards, batch_dones = replay_buffer.sample(BATCH_SIZE_FOR_TEST)

    print(f"Sampled batch_obs shape: {batch_obs.shape}") # (Batch, NumAgents, C, H, W)
    print(f"Sampled batch_states shape: {batch_states.shape}") # (Batch, C_state, H, W)

    # Chọn một mẫu từ batch để visualize (ví dụ: mẫu đầu tiên, agent đầu tiên)
    sample_idx_in_batch = 0
    agent_to_visualize_idx = 0 # Hoặc current_robot_idx bạn đã dùng trước đó

    # Tensor obs và next_obs của agent được chọn từ mẫu được chọn
    # Chuyển từ tensor PyTorch về numpy để imshow
    obs_to_visualize = batch_obs[sample_idx_in_batch, agent_to_visualize_idx].cpu().numpy()
    next_obs_to_visualize = batch_next_obs[sample_idx_in_batch, agent_to_visualize_idx].cpu().numpy()

    # Tensor global state và next_global_state của mẫu được chọn
    state_to_visualize = batch_states[sample_idx_in_batch].cpu().numpy()
    next_state_to_visualize = batch_next_states[sample_idx_in_batch].cpu().numpy()


    # --- Visualize Observation (Robot-Specific) ---
    channel_names_obs = [
        "Map", "Urgency", "Start Pos", "Target Pos",
        "Robot Carrying", "Other Robots", "Current Robot", "Carried Pkg Target"
    ]
    n_channels_obs = obs_to_visualize.shape[0]

    if n_channels_obs != len(channel_names_obs):
        print(f"Warning: Number of channels in obs_to_visualize ({n_channels_obs}) "
              f"does not match len(channel_names_obs) ({len(channel_names_obs)}). "
              "Channel names might be incorrect.")
        # Điều chỉnh channel_names_obs nếu cần hoặc kiểm tra convert_state
        channel_names_obs = [f"Obs Channel {i}" for i in range(n_channels_obs)]


    fig_obs, axes_obs = plt.subplots(2, n_channels_obs, figsize=(3 * n_channels_obs, 3 * 2 + 1))
    fig_obs.suptitle(f"Sampled Observation (Agent {agent_to_visualize_idx}, Sample {sample_idx_in_batch})", fontsize=16)

    for i in range(n_channels_obs):
        # Plot obs_t
        ax_t = axes_obs[0, i]
        im_t = ax_t.imshow(obs_to_visualize[i], cmap='viridis', interpolation='nearest', vmin=obs_to_visualize[i].min(), vmax=max(0.1, obs_to_visualize[i].max()))
        if i == 0:
            ax_t.set_ylabel("Obs (t)", fontsize=12)
        ax_t.set_title(channel_names_obs[i])
        ax_t.axis('off')
        plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

        # Plot obs_t+1
        ax_t1 = axes_obs[1, i]
        im_t1 = ax_t1.imshow(next_obs_to_visualize[i], cmap='viridis', interpolation='nearest', vmin=next_obs_to_visualize[i].min(), vmax=max(0.1, next_obs_to_visualize[i].max()))
        if i == 0:
            ax_t1.set_ylabel("Next Obs (t+1)", fontsize=12)
        ax_t1.axis('off')
        plt.colorbar(im_t1, ax=ax_t1, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()


    # --- Visualize Global State ---
    channel_names_state = [
        "Map", "Urgency", "Start Pos", "Target Pos",
        "Robot Carrying", "All Robots"
    ]
    n_channels_state = state_to_visualize.shape[0]

    if n_channels_state != len(channel_names_state):
        print(f"Warning: Number of channels in state_to_visualize ({n_channels_state}) "
              f"does not match len(channel_names_state) ({len(channel_names_state)}). "
              "Channel names might be incorrect.")
        channel_names_state = [f"State Channel {i}" for i in range(n_channels_state)]


    fig_state, axes_state = plt.subplots(2, n_channels_state, figsize=(3 * n_channels_state, 3 * 2 + 1))
    fig_state.suptitle(f"Sampled Global State (Sample {sample_idx_in_batch})", fontsize=16)

    for i in range(n_channels_state):
        # Plot state_t
        ax_s_t = axes_state[0, i]
        im_s_t = ax_s_t.imshow(state_to_visualize[i], cmap='viridis', interpolation='nearest', vmin=state_to_visualize[i].min(), vmax=max(0.1, state_to_visualize[i].max()))
        if i == 0:
            ax_s_t.set_ylabel("State (t)", fontsize=12)
        ax_s_t.set_title(channel_names_state[i])
        ax_s_t.axis('off')
        plt.colorbar(im_s_t, ax=ax_s_t, fraction=0.046, pad=0.04)

        # Plot state_t+1
        ax_s_t1 = axes_state[1, i]
        im_s_t1 = ax_s_t1.imshow(next_state_to_visualize[i], cmap='viridis', interpolation='nearest', vmin=next_state_to_visualize[i].min(), vmax=max(0.1, next_state_to_visualize[i].max()))
        if i == 0:
            ax_s_t1.set_ylabel("Next State (t+1)", fontsize=12)
        ax_s_t1.axis('off')
        plt.colorbar(im_s_t1, ax=ax_s_t1, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

else:
    print(f"Buffer does not have enough samples ({len(replay_buffer)}) to test with batch size {BATCH_SIZE_FOR_TEST}.")