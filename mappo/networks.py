import numpy as np

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


def convert_global_state_to_tensor(state_dict, persistent_packages, state_tensor_shape):
    """
    Converts the global state dictionary to a tensor for QMIX.
    Relies on `persistent_packages` for all package information.
    The `packages` key in `state_dict` (if present) is ignored for package data.

    Args:
        state_dict (dict): The raw environment state dictionary.
                           Expected keys: "map", "robots", "time_step".
        persistent_packages (dict): Dictionary tracking all active packages.
                                    Positions are 0-indexed.
                                    Example entry:
                                    { pkg_id: {'start_pos': (r,c), 'target_pos': (r,c),
                                                'status': 'waiting'/'in_transit',
                                                'start_time': ts, 'deadline': dl, 'id': pkg_id} }
        state_tensor_shape (tuple): Tuple (num_channels, n_rows, n_cols) for the output state tensor.
        max_time_steps (int): Maximum time steps in an episode for normalization.

    Returns:
        np.ndarray: The global state tensor with shape specified by state_tensor_shape.
        float: Normalized current time step (scalar feature).
    """
    num_channels_out, n_rows, n_cols = state_tensor_shape
    
    spatial_tensor = np.zeros((num_channels_out, n_rows, n_cols), dtype=np.float32)

    CH_IDX_MAP_OBSTACLES = 0
    CH_IDX_ROBOT_POSITIONS = 1
    CH_IDX_ROBOT_CARRYING_STATUS = 2
    CH_IDX_PKG_WAITING_START_POS = 3
    CH_IDX_PKG_WAITING_TARGET_POS = 4
    CH_IDX_PKG_IN_TRANSIT_TARGET_POS = 5
    CH_IDX_PKG_WAITING_URGENCY = 6

    # --- Channel: Map Obstacles (Centering/Cropping Logic) ---
    if CH_IDX_MAP_OBSTACLES < num_channels_out:
        game_map_from_state = np.array(state_dict["map"])
        map_rows_src, map_cols_src = game_map_from_state.shape

        src_r_start = (map_rows_src - n_rows) // 2 if map_rows_src > n_rows else 0
        src_c_start = (map_cols_src - n_cols) // 2 if map_cols_src > n_cols else 0
        
        rows_to_copy_from_src = min(map_rows_src, n_rows)
        cols_to_copy_from_src = min(map_cols_src, n_cols)

        map_section_to_copy = game_map_from_state[
            src_r_start : src_r_start + rows_to_copy_from_src,
            src_c_start : src_c_start + cols_to_copy_from_src
        ]
        
        target_r_offset = (n_rows - map_section_to_copy.shape[0]) // 2
        target_c_offset = (n_cols - map_section_to_copy.shape[1]) // 2
            
        spatial_tensor[
            CH_IDX_MAP_OBSTACLES,
            target_r_offset : target_r_offset + map_section_to_copy.shape[0],
            target_c_offset : target_c_offset + map_section_to_copy.shape[1]
        ] = map_section_to_copy

    # --- Current Time (Scalar Feature) ---
    current_time = state_dict["time_step"]

    # --- Channels: Robot Positions and Carrying Status (from state_dict['robots']) ---
    if 'robots' in state_dict and state_dict['robots'] is not None:
        for r_data in state_dict['robots']:
            # r_data: (pos_r_1idx, pos_c_1idx, carrying_package_id)
            r_idx, c_idx = int(r_data[0]) - 1, int(r_data[1]) - 1 # Convert to 0-indexed
            carried_pkg_id = r_data[2]

            if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols: # Boundary check
                if CH_IDX_ROBOT_POSITIONS < num_channels_out:
                    spatial_tensor[CH_IDX_ROBOT_POSITIONS, r_idx, c_idx] = 1.0
                
                if carried_pkg_id != 0 and CH_IDX_ROBOT_CARRYING_STATUS < num_channels_out:
                    spatial_tensor[CH_IDX_ROBOT_CARRYING_STATUS, r_idx, c_idx] = 1.0

    # --- Process persistent_packages for ALL package-related channels ---
    # Note: state_dict['packages'] is NOT used here.
    for pkg_id, pkg_data in persistent_packages.items():
        start_pos = pkg_data['start_pos']   # Expected (r, c) 0-indexed
        target_pos = pkg_data['target_pos'] # Expected (r, c) 0-indexed
        status = pkg_data['status']
        pkg_start_time = pkg_data['start_time']
        pkg_deadline = pkg_data['deadline']
        
        # Process only if package is active based on its start_time
        if current_time >= pkg_start_time:
            if status == 'waiting':
                # Channel: Waiting Packages' Start Positions
                if CH_IDX_PKG_WAITING_START_POS < num_channels_out:
                    if 0 <= start_pos[0] < n_rows and 0 <= start_pos[1] < n_cols: # Boundary check
                        spatial_tensor[CH_IDX_PKG_WAITING_START_POS, start_pos[0], start_pos[1]] = 1.0

                # Channel: Urgency of Waiting Packages
                if CH_IDX_PKG_WAITING_URGENCY < num_channels_out:
                    if 0 <= start_pos[0] < n_rows and 0 <= start_pos[1] < n_cols: # Boundary check
                        urgency = 0.0
                        if pkg_deadline > pkg_start_time: 
                            urgency = min(1.0, max(0.0, (current_time - pkg_start_time) / (pkg_deadline - pkg_start_time)))
                        elif pkg_deadline == pkg_start_time: 
                            urgency = 1.0 # Deadline is now or passed if current_time >= pkg_start_time
                        # Use max in case multiple packages share the same start_pos
                        spatial_tensor[CH_IDX_PKG_WAITING_URGENCY, start_pos[0], start_pos[1]] = \
                            max(spatial_tensor[CH_IDX_PKG_WAITING_URGENCY, start_pos[0], start_pos[1]], urgency)

                # Channel: Waiting Packages' Target Positions
                if CH_IDX_PKG_WAITING_TARGET_POS < num_channels_out:
                    if 0 <= target_pos[0] < n_rows and 0 <= target_pos[1] < n_cols: # Boundary check
                        spatial_tensor[CH_IDX_PKG_WAITING_TARGET_POS, target_pos[0], target_pos[1]] = \
                            max(spatial_tensor[CH_IDX_PKG_WAITING_TARGET_POS, target_pos[0], target_pos[1]], 1.0)
            
            elif status == 'in_transit':
                # Channel: In-Transit Packages' Target Positions
                if CH_IDX_PKG_IN_TRANSIT_TARGET_POS < num_channels_out:
                    if 0 <= target_pos[0] < n_rows and 0 <= target_pos[1] < n_cols: # Boundary check
                        spatial_tensor[CH_IDX_PKG_IN_TRANSIT_TARGET_POS, target_pos[0], target_pos[1]] = \
                            max(spatial_tensor[CH_IDX_PKG_IN_TRANSIT_TARGET_POS, target_pos[0], target_pos[1]], 1.0)
                
    return spatial_tensor