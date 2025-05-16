# helper.py
import os
import numpy as np
import torch

def convert_observation(env_state_dict, persistent_packages_for_env, current_robot_idx):
    """
    Tạo observation dạng multi-channel cho 1 robot.
    Channels:
        0: Obstacle map
        1: Vị trí robot hiện tại
        2: Vị trí các robot khác
        3: Vị trí start của các package 'waiting'
        4: Vị trí target của các package 'active' (waiting hoặc in_transit)
        5: Vị trí target của package mà robot này đang cầm (nếu có)
    """
    num_channels = 6
    grid = np.array(env_state_dict['map'], dtype=np.float32)
    n_rows, n_cols = grid.shape
    obs = np.zeros((num_channels, n_rows, n_cols), dtype=np.float32)

    # Channel 0: Obstacle map
    obs[0] = grid

    # Kiểm tra robot index hợp lệ
    if not (0 <= current_robot_idx < len(env_state_dict['robots'])):
        return obs

    robots = env_state_dict['robots']
    my_r, my_c, my_pkg = [int(x) for x in robots[current_robot_idx]]
    my_r -= 1; my_c -= 1  # 0-indexed

    # Channel 1: Vị trí robot hiện tại
    if 0 <= my_r < n_rows and 0 <= my_c < n_cols:
        obs[1, my_r, my_c] = 1.0

    # Channel 2: Vị trí các robot khác
    for i, (r, c, _) in enumerate(robots):
        if i == current_robot_idx: continue
        r, c = int(r)-1, int(c)-1
        if 0 <= r < n_rows and 0 <= c < n_cols:
            obs[2, r, c] = 1.0

    # Channel 3, 4, 5: Package info
    t = env_state_dict['time_step']
    for pkg_id, pkg in persistent_packages_for_env.items():
        # Channel 3: Start pos của package 'waiting'
        if pkg['status'] == 'waiting' and pkg['start_time'] <= t:
            sr, sc = pkg['start_pos']
            if 0 <= sr < n_rows and 0 <= sc < n_cols:
                obs[3, sr, sc] = 1.0
        # Channel 4: Target pos của package 'active'
        if (pkg['status'] == 'waiting' and pkg['start_time'] <= t) or pkg['status'] == 'in_transit':
            tr, tc = pkg['target_pos']
            if 0 <= tr < n_rows and 0 <= tc < n_cols:
                obs[4, tr, tc] = 1.0

    # Channel 5: Target pos của package mà robot này đang cầm
    if my_pkg != 0 and my_pkg in persistent_packages_for_env:
        pkg = persistent_packages_for_env[my_pkg]
        if pkg['status'] == 'in_transit':
            tr, tc = pkg['target_pos']
            if 0 <= tr < n_rows and 0 <= tc < n_cols:
                obs[5, tr, tc] = 1.0

    return obs

def generate_vector_features(env_state_dict, persistent_packages_for_env, current_robot_idx,
                            max_time_steps,
                            max_other_robots_to_observe=100, max_packages_to_observe=100):
    """
    Sinh vector đặc trưng phi không gian cho 1 robot.
    """
    n_rows, n_cols = np.array(env_state_dict['map'], dtype=np.float32).shape
    robots = env_state_dict['robots']
    t = env_state_dict['time_step']

    # Định nghĩa số chiều cho từng phần
    my_feat = 6
    other_feat = 5
    pkg_feat = 5
    time_feat = 1

    # Nếu robot index không hợp lệ, trả về vector 0
    total_len = my_feat + max_other_robots_to_observe * other_feat + max_packages_to_observe * pkg_feat + time_feat
    if not (0 <= current_robot_idx < len(robots)):
        return np.zeros(total_len, dtype=np.float32)

    # 1. Thông tin robot hiện tại
    my_r, my_c, my_pkg = [int(x) for x in robots[current_robot_idx]]
    my_r -= 1; my_c -= 1
    is_carrying = 1.0 if my_pkg != 0 else 0.0
    feat = [
        my_r / n_rows,
        my_c / n_cols,
        is_carrying
    ]
    # Nếu đang cầm package
    if is_carrying and my_pkg in persistent_packages_for_env:
        pkg = persistent_packages_for_env[my_pkg]
        if pkg['status'] == 'in_transit':
            tr, tc = pkg['target_pos']
            deadline = pkg['deadline']
            feat += [
                (tr - my_r) / n_rows,
                (tc - my_c) / n_cols,
                max(0, deadline - t) / max_time_steps if max_time_steps > 0 else 0.0
            ]
        else:
            feat += [0.0, 0.0, 0.0]
    else:
        feat += [0.0, 0.0, 0.0]

    # 2. Thông tin các robot khác
    others = []
    for i, (r, c, pkg_id) in enumerate(robots):
        if i == current_robot_idx: continue
        r, c, pkg_id = int(r)-1, int(c)-1, int(pkg_id)
        is_c = 1.0 if pkg_id != 0 else 0.0
        other = [
            (r - my_r) / n_rows,
            (c - my_c) / n_cols,
            is_c
        ]
        if is_c and pkg_id in persistent_packages_for_env:
            pkg = persistent_packages_for_env[pkg_id]
            if pkg['status'] == 'in_transit':
                tr, tc = pkg['target_pos']
                other += [
                    (tr - r) / n_rows,
                    (tc - c) / n_cols
                ]
            else:
                other += [0.0, 0.0]
        else:
            other += [0.0, 0.0]
        others.append(other)
    # Sắp xếp theo khoảng cách tới robot hiện tại
    others.sort(key=lambda x: x[0]**2 + x[1]**2)
    for i in range(max_other_robots_to_observe):
        feat += others[i] if i < len(others) else [0.0]*other_feat

    # 3. Thông tin các package 'waiting'
    pkgs = []
    for pkg_id, pkg in persistent_packages_for_env.items():
        if pkg['status'] == 'waiting' and pkg['start_time'] <= t:
            sr, sc = pkg['start_pos']
            tr, tc = pkg['target_pos']
            deadline = pkg['deadline']
            pkgs.append([
                (sr - my_r) / n_rows,
                (sc - my_c) / n_cols,
                (tr - my_r) / n_rows,
                (tc - my_c) / n_cols,
                max(0, deadline - t) / max_time_steps if max_time_steps > 0 else 0.0
            ])
    # Sắp xếp theo deadline và khoảng cách
    pkgs.sort(key=lambda x: (x[4], x[0]**2 + x[1]**2))
    for i in range(max_packages_to_observe):
        feat += pkgs[i] if i < len(pkgs) else [0.0]*pkg_feat

    # 4. Thời gian toàn cục (chuẩn hóa)
    feat.append(t / max_time_steps if max_time_steps > 0 else 0.0)

    return np.array(feat, dtype=np.float32)

def convert_global_state(env_state_dict, persistent_packages_for_env,
                                max_time_steps,
                                max_robots_in_state=100, max_packages_in_state=100):
    """
    Sinh global state (spatial + vector) cho Critic.
    """
    # --- Spatial ---
    num_map_channels = 4
    n_rows, n_cols = np.array(env_state_dict['map'], dtype=np.float32).shape
    
    global_map = np.zeros((num_map_channels, n_rows, n_cols), dtype=np.float32)
    global_map[0] = np.array(env_state_dict['map'], dtype=np.float32)  # Obstacles

    # Channel 1: All robot positions
    for r, c, _ in env_state_dict['robots']:
        r0, c0 = int(r)-1, int(c)-1
        if 0 <= r0 < n_rows and 0 <= c0 < n_cols:
            global_map[1, r0, c0] = 1.0

    t = env_state_dict['time_step']
    for pkg in persistent_packages_for_env.values():
        # Channel 2: waiting package start
        if pkg['status'] == 'waiting' and pkg['start_time'] <= t:
            sr, sc = pkg['start_pos']
            if 0 <= sr < n_rows and 0 <= sc < n_cols:
                global_map[2, sr, sc] = 1.0
        # Channel 3: active package target
        if (pkg['status'] == 'waiting' and pkg['start_time'] <= t) or pkg['status'] == 'in_transit':
            tr, tc = pkg['target_pos']
            if 0 <= tr < n_rows and 0 <= tc < n_cols:
                global_map[3, tr, tc] = 1.0

    # --- Vector ---
    vec = []
    # 1. Robots (padded)
    for i in range(max_robots_in_state):
        if i < len(env_state_dict['robots']):
            r, c, carried = env_state_dict['robots'][i]
            r0, c0 = int(r)-1, int(c)-1
            is_carrying = 1.0 if carried != 0 else 0.0
            vec += [r0/n_rows, c0/n_cols, is_carrying]
            if is_carrying and carried in persistent_packages_for_env:
                pkg = persistent_packages_for_env[carried]
                if pkg['status'] == 'in_transit':
                    tr, tc = pkg['target_pos']
                    deadline = pkg['deadline']
                    vec += [tr/n_rows, tc/n_cols, max(0, deadline-t)/max_time_steps if max_time_steps > 0 else 0.0]
                else:
                    vec += [0.0, 0.0, 0.0]
            else:
                vec += [0.0, 0.0, 0.0]
        else:
            vec += [0.0]*6

    # 2. Active packages (padded)
    pkgs = []
    for pkg in persistent_packages_for_env.values():
        is_active = (pkg['status'] == 'waiting' and pkg['start_time'] <= t) or pkg['status'] == 'in_transit'
        if is_active:
            pkgs.append(pkg)
    pkgs = sorted(pkgs, key=lambda p: p['id'])
    for i in range(max_packages_in_state):
        if i < len(pkgs):
            pkg = pkgs[i]
            sr, sc = pkg['start_pos']
            tr, tc = pkg['target_pos']
            deadline = pkg['deadline']
            status = pkg['status']
            # start pos (nếu waiting), target pos, deadline, status, carrier_id_norm
            if status == 'waiting':
                vec += [sr/n_rows, sc/n_cols]
            else:
                vec += [0.0, 0.0]
            vec += [tr/n_rows, tc/n_cols]
            vec += [max(0, deadline-t)/max_time_steps if max_time_steps > 0 else 0.0]
            vec += [0.0 if status == 'waiting' else 1.0]
            carrier_id_norm = -1.0
            if status == 'in_transit':
                for ridx, rdata in enumerate(env_state_dict['robots']):
                    if rdata[2] == pkg['id']:
                        carrier_id_norm = ridx/(max_robots_in_state-1) if max_robots_in_state > 1 else 0.0
                        break
            vec += [carrier_id_norm]
        else:
            vec += [0.0]*7

    # 3. Global time
    vec.append(t/max_time_steps if max_time_steps > 0 else 0.0)
    return global_map, np.array(vec, dtype=np.float32)

def compute_shaped_rewards(
    global_reward,
    prev_env_state_dict,
    current_env_state_dict,
    actions_taken_for_all_agents,
    persistent_packages_at_prev_state,
    num_agents,
):
    """
    Computes shaped rewards for each agent based on transitions and intended actions.
    Returns: tổng shaped reward (float), và shaped reward từng agent (np.array)
    """
    # --- Shaping Constants ---
    SHAPING_SUCCESSFUL_PICKUP = 0.5
    SHAPING_SUCCESSFUL_DELIVERY_ON_TIME = 2.0
    SHAPING_SUCCESSFUL_DELIVERY_LATE = 0.2
    SHAPING_MOVED_CLOSER_TO_TARGET = 0.02
    SHAPING_WASTED_PICKUP_ATTEMPT = -0.1
    SHAPING_WASTED_DROP_ATTEMPT = -0.1
    SHAPING_COLLISION_OR_STUCK = -0.05
    SHAPING_IDLE_WITH_AVAILABLE_TASKS = -0.02
    SHAPING_MOVED_AWAY_FROM_TARGET = -0.02

    shaped_rewards = np.zeros(num_agents, dtype=np.float32)
    current_time = int(current_env_state_dict['time_step'])

    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    for agent_idx in range(num_agents):
        prev_r, prev_c, prev_pkg = [int(x) for x in prev_env_state_dict['robots'][agent_idx]]
        curr_r, curr_c, curr_pkg = [int(x) for x in current_env_state_dict['robots'][agent_idx]]
        prev_r -= 1; prev_c -= 1; curr_r -= 1; curr_c -= 1
        move_str, pkg_op_str = actions_taken_for_all_agents[agent_idx]
        pkg_op = int(pkg_op_str)

        # 1. Nhặt/thả thành công
        if prev_pkg == 0 and curr_pkg != 0:
            shaped_rewards[agent_idx] += SHAPING_SUCCESSFUL_PICKUP
        elif prev_pkg != 0 and curr_pkg == 0:
            dropped_pkg = prev_pkg
            if dropped_pkg in persistent_packages_at_prev_state:
                pkg_info = persistent_packages_at_prev_state[dropped_pkg]
                if (curr_r, curr_c) == pkg_info['target_pos']:
                    if current_time <= pkg_info['deadline']:
                        shaped_rewards[agent_idx] += SHAPING_SUCCESSFUL_DELIVERY_ON_TIME
                    else:
                        shaped_rewards[agent_idx] += SHAPING_SUCCESSFUL_DELIVERY_LATE

        # 2. Phạt hành động lãng phí
        if pkg_op == 1:  # Pick
            if prev_pkg != 0:
                shaped_rewards[agent_idx] += SHAPING_WASTED_PICKUP_ATTEMPT
            elif curr_pkg == 0:
                can_pickup = any(
                    pkg['status'] == 'waiting' and
                    pkg['start_time'] <= prev_env_state_dict['time_step'] and
                    pkg['start_pos'] == (curr_r, curr_c)
                    for pkg in persistent_packages_at_prev_state.values()
                )
                if not can_pickup:
                    shaped_rewards[agent_idx] += SHAPING_WASTED_PICKUP_ATTEMPT
        elif pkg_op == 2:  # Drop
            if prev_pkg == 0:
                shaped_rewards[agent_idx] += SHAPING_WASTED_DROP_ATTEMPT
            elif curr_pkg != 0:
                if prev_pkg in persistent_packages_at_prev_state:
                    pkg_info = persistent_packages_at_prev_state[prev_pkg]
                    if (curr_r, curr_c) != pkg_info['target_pos']:
                        shaped_rewards[agent_idx] += SHAPING_WASTED_DROP_ATTEMPT

        # 3. Di chuyển
        moved = (prev_r, prev_c) != (curr_r, curr_c)
        intended_move = move_str != 'S'
        if intended_move and not moved:
            shaped_rewards[agent_idx] += SHAPING_COLLISION_OR_STUCK

        # Tính mục tiêu di chuyển
        target_pos = None
        if prev_pkg != 0 and prev_pkg in persistent_packages_at_prev_state:
            target_pos = persistent_packages_at_prev_state[prev_pkg]['target_pos']
        else:
            # Gói waiting gần nhất
            waiting_pkgs = [
                pkg for pkg in persistent_packages_at_prev_state.values()
                if pkg['status'] == 'waiting' and pkg['start_time'] <= prev_env_state_dict['time_step']
            ]
            if waiting_pkgs:
                target_pos = min(
                    (pkg['start_pos'] for pkg in waiting_pkgs),
                    key=lambda pos: manhattan_distance((prev_r, prev_c), pos)
                )
        if target_pos and moved:
            dist_before = manhattan_distance((prev_r, prev_c), target_pos)
            dist_after = manhattan_distance((curr_r, curr_c), target_pos)
            if dist_after < dist_before:
                shaped_rewards[agent_idx] += SHAPING_MOVED_CLOSER_TO_TARGET
            elif dist_after > dist_before:
                shaped_rewards[agent_idx] += SHAPING_MOVED_AWAY_FROM_TARGET

        # 4. Phạt đứng yên không cần thiết
        if not moved and move_str == 'S' and prev_pkg == 0:
            idle_nearby = any(
                pkg['status'] == 'waiting' and
                pkg['start_time'] <= prev_env_state_dict['time_step'] and
                manhattan_distance((prev_r, prev_c), pkg['start_pos']) <= 3
                for pkg in persistent_packages_at_prev_state.values()
            )
            if idle_nearby:
                shaped_rewards[agent_idx] += SHAPING_IDLE_WITH_AVAILABLE_TASKS

    return global_reward + shaped_rewards.sum() 

def save_mappo_model(actor, critic, path_prefix="models/mappo"):
    abs_prefix = os.path.abspath(path_prefix)
    dir_name = os.path.dirname(abs_prefix)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(actor.state_dict(), f"{abs_prefix}_actor.pt")
    torch.save(critic.state_dict(), f"{abs_prefix}_critic.pt")
    print(f"MAPPO models saved with prefix {path_prefix}")

def load_mappo_model(actor, critic, path_prefix="models/mappo", device="cpu"):
    actor_path = f"{path_prefix}_actor.pt"
    critic_path = f"{path_prefix}_critic.pt"
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        print(f"MAPPO models loaded from prefix {path_prefix}")
        return True
    print(f"Could not find MAPPO models at prefix {path_prefix}")
    return False