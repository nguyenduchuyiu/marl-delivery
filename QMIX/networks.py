import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNNAgent(nn.Module):
    """
    Mạng Q-value cá nhân cho mỗi tác nhân, sử dụng RNN (GRUCell).
    Đầu vào: quan sát cục bộ (và có thể là hành động trước đó, ID tác nhân).
    Đầu ra: Q-values cho mỗi hành động có thể có của tác nhân.
    """
    def __init__(self, spatial_obs_shape, vector_obs_dim, rnn_hidden_dim, n_actions=15,
                 cnn_channels_out=64, cnn_mlp_hidden_dim=128, vector_mlp_hidden_dim=128, args=None):
        """
        RNNAgent xử lý cả spatial và vector observations.

        Args:
            spatial_obs_shape (tuple): Shape của spatial observation (C, H, W).
            vector_obs_dim (int): Dimension của vector observation.
            rnn_hidden_dim (int): Kích thước lớp ẩn của GRU.
            n_actions (int): Số lượng hành động.
            cnn_channels_out (int): Số kênh đầu ra của lớp CNN cuối cùng.
            cnn_mlp_hidden_dim (int): Kích thước lớp ẩn MLP sau CNN.
            vector_mlp_hidden_dim (int): Kích thước lớp ẩn MLP cho vector obs.
            args: Các tham số khác.
        """
        super(RNNAgent, self).__init__()
        self.args = args
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions

        # --- CNN Branch for Spatial Observations ---
        self.spatial_channels_in = spatial_obs_shape[0]
        # Giả sử kernel_size=3, stride=1, padding=1 không thay đổi kích thước H, W đáng kể
        # Hoặc sử dụng AdaptiveAvgPool2d để có kích thước cố định
        self.conv1 = nn.Conv2d(self.spatial_channels_in, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, cnn_channels_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels_out)
        # self.conv3 = nn.Conv2d(64, cnn_channels_out, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(cnn_channels_out)
        
        # Sử dụng AdaptiveAvgPool2d để có kích thước đầu ra cố định từ CNN
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Ví dụ: output 4x4
        self.cnn_flattened_dim = cnn_channels_out * 4 * 4
        self.cnn_fc = nn.Linear(self.cnn_flattened_dim, cnn_mlp_hidden_dim)


        # --- MLP Branch for Vector Observations ---
        self.vector_fc1 = nn.Linear(vector_obs_dim, vector_mlp_hidden_dim)
        # self.vector_fc2 = nn.Linear(vector_mlp_hidden_dim, vector_mlp_hidden_dim // 2)

        # --- Combined Features to GRU ---
        # combined_input_dim = self.cnn_flattened_dim + (vector_mlp_hidden_dim // 2)
        combined_input_dim = cnn_mlp_hidden_dim + vector_mlp_hidden_dim
        
        self.rnn = nn.GRUCell(combined_input_dim, rnn_hidden_dim)
        self.fc_q = nn.Linear(rnn_hidden_dim, n_actions) # Output Q-values

    def init_hidden(self):
        # Khởi tạo trạng thái ẩn cho RNN (trên cùng thiết bị với model)
        return self.conv1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, spatial_obs, vector_obs, hidden_state):
        # spatial_obs: (batch_size, C, H, W)
        # vector_obs: (batch_size, vector_obs_dim)
        # hidden_state shape: (batch_size, rnn_hidden_dim)

        # CNN path
        x_spatial = F.relu(self.bn1(self.conv1(spatial_obs)))
        x_spatial = F.relu(self.bn2(self.conv2(x_spatial)))
        # x_spatial = F.relu(self.bn3(self.conv3(x_spatial))) # Nếu có conv3
        x_spatial = self.adaptive_pool(x_spatial)
        x_spatial_flat = x_spatial.reshape(x_spatial.size(0), -1)
        x_spatial_processed = F.relu(self.cnn_fc(x_spatial_flat))

        # Vector MLP path
        x_vector_processed = F.relu(self.vector_fc1(vector_obs))
        # x_vector_processed = F.relu(self.vector_fc2(x_vector)) # Nếu có vector_fc2

        # Concatenate processed features
        combined_features = torch.cat((x_spatial_processed, x_vector_processed), dim=1)

        # GRU
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(combined_features, h_in)
        
        q_values = self.fc_q(h)
        return q_values, h

class QMixer(nn.Module):
    """
    Mạng Trộn QMIX.
    Kết hợp Q-values từ các tác nhân cá nhân thành Q_tot.
    Sử dụng hypernetworks để tạo trọng số và bias cho mạng trộn,
    đảm bảo tính đơn điệu (monotonicity).
    """
    def __init__(self, n_agents, 
                 global_spatial_state_shape, global_vector_state_dim, 
                 mixing_embed_dim, 
                 cnn_channels_out=64, cnn_mlp_hidden_dim=128, 
                 vector_mlp_hidden_dim=128, hypernet_embed=64, args=None):
        super(QMixer, self).__init__()
        self.args = args
        self.n_agents = n_agents
        self.mixing_embed_dim = mixing_embed_dim # Kích thước embedding cho Q-values trong mạng trộn
        self.hypernet_embed = hypernet_embed  # Store for use in hypernet layers

        # --- Processor for Global State (Spatial + Vector) ---
        self.global_spatial_channels_in = global_spatial_state_shape[0]
        self.state_conv1 = nn.Conv2d(self.global_spatial_channels_in, 32, kernel_size=3, stride=1, padding=1)
        self.state_bn1 = nn.BatchNorm2d(32)
        self.state_conv2 = nn.Conv2d(32, cnn_channels_out, kernel_size=3, stride=1, padding=1)
        self.state_bn2 = nn.BatchNorm2d(cnn_channels_out)
        self.state_adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.state_cnn_flattened_dim = cnn_channels_out * 4 * 4
        self.state_cnn_fc = nn.Linear(self.state_cnn_flattened_dim, cnn_mlp_hidden_dim)

        self.state_vector_fc1 = nn.Linear(global_vector_state_dim, vector_mlp_hidden_dim)
        
        # Kích thước của state sau khi xử lý, dùng làm đầu vào cho hypernetworks
        self.processed_state_dim = cnn_mlp_hidden_dim + vector_mlp_hidden_dim

        # Hypernetwork cho trọng số của lớp trộn thứ nhất
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.processed_state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.n_agents * self.mixing_embed_dim)
        )
        # Hypernetwork cho bias của lớp trộn thứ nhất
        self.hyper_b1 = nn.Linear(self.processed_state_dim, self.mixing_embed_dim)

        # Hypernetwork cho trọng số của lớp trộn thứ hai
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.processed_state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.mixing_embed_dim) # w2 là vector, không phải ma trận
        )
        # Hypernetwork cho bias của lớp trộn thứ hai (scalar cho Q_tot)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.processed_state_dim, self.mixing_embed_dim), # Lớp trung gian
            nn.ReLU(),
            nn.Linear(self.mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, global_spatial_state, global_vector_state):
        # agent_qs: Q-values của các tác nhân, shape (batch_size, seq_len, n_agents) hoặc (batch_size, n_agents)
        # global_spatial_state: (batch_size, seq_len, C_global, H, W) hoặc (batch_size, C_global, H, W)
        # global_vector_state: (batch_size, seq_len, global_vector_dim) hoặc (batch_size, global_vector_dim)

        original_shape_qs = agent_qs.shape
        bs = original_shape_qs[0]
        seq_len = 1
        if len(original_shape_qs) == 3: # (bs, seq_len, N)
            seq_len = original_shape_qs[1]
            agent_qs = agent_qs.reshape(bs * seq_len, self.n_agents)
            current_c_global = self.global_spatial_state_shape[0]
            current_h_global = self.global_spatial_state_shape[1]
            current_w_global = self.global_spatial_state_shape[2]
            global_spatial_state = global_spatial_state.reshape(bs * seq_len, current_c_global, current_h_global, current_w_global)
            global_vector_state = global_vector_state.reshape(bs * seq_len, self.global_vector_state_dim)

        s_spatial = global_spatial_state
        s_vector = global_vector_state

        s_spatial_proc = F.relu(self.state_bn1(self.state_conv1(s_spatial)))
        s_spatial_proc = F.relu(self.state_bn2(self.state_conv2(s_spatial_proc)))
        s_spatial_proc = self.state_adaptive_pool(s_spatial_proc)
        s_spatial_flat = s_spatial_proc.reshape(s_spatial_proc.size(0), -1)
        s_spatial_out = F.relu(self.state_cnn_fc(s_spatial_flat))
        s_vector_out = F.relu(self.state_vector_fc1(s_vector))
        processed_state = torch.cat((s_spatial_out, s_vector_out), dim=1)

        # Lớp trộn thứ nhất
        w1_val = self.hyper_w1(processed_state)
        w1 = torch.abs(w1_val).view(-1, self.n_agents, self.mixing_embed_dim)

        b1_val = self.hyper_b1(processed_state)
        b1 = b1_val.view(-1, 1, self.mixing_embed_dim)

        agent_qs_reshaped = agent_qs.unsqueeze(1)

        # Perform BMM1
        hidden_bmm1_out = torch.bmm(agent_qs_reshaped, w1)
        hidden = F.elu(hidden_bmm1_out + b1)

        # Lớp trộn thứ hai
        w2_val = self.hyper_w2(processed_state)
        w2 = torch.abs(w2_val).view(-1, self.mixing_embed_dim, 1)

        b2 = self.hyper_b2(processed_state)

        # Perform BMM2
        q_total_bmm2_out = torch.bmm(hidden, w2)
        
        # Corrected addition using unsqueeze to ensure proper broadcasting
        q_total_before_squeeze = q_total_bmm2_out + b2.unsqueeze(1) 
        
        q_total = q_total_before_squeeze.squeeze(-1)

        if len(original_shape_qs) == 3:
            q_total = q_total.view(bs, seq_len, 1)
        else:
            q_total = q_total.view(bs, 1)
            
        return q_total

class ReplayBuffer:
    def __init__(self, capacity, episode_limit, n_agents,
                 spatial_obs_shape, vector_obs_dim,
                 global_spatial_state_shape, global_vector_state_dim,
                 n_actions, args):
        self.capacity = capacity
        self.episode_limit = episode_limit
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.args = args

        # Shapes
        self.spatial_obs_shape = spatial_obs_shape # (C_obs, H, W)
        self.vector_obs_dim = vector_obs_dim
        self.global_spatial_state_shape = global_spatial_state_shape # (C_global, H, W)
        self.global_vector_state_dim = global_vector_state_dim

        # Initialize buffers
        self.buffer_spatial_obs = np.zeros((capacity, episode_limit, n_agents, *spatial_obs_shape), dtype=np.float32)
        self.buffer_vector_obs = np.zeros((capacity, episode_limit, n_agents, vector_obs_dim), dtype=np.float32)
        
        self.buffer_global_spatial_state = np.zeros((capacity, episode_limit, *global_spatial_state_shape), dtype=np.float32)
        self.buffer_global_vector_state = np.zeros((capacity, episode_limit, global_vector_state_dim), dtype=np.float32)
        
        self.buffer_actions = np.zeros((capacity, episode_limit, n_agents, 1), dtype=np.int64)
        self.buffer_rewards = np.zeros((capacity, episode_limit, 1), dtype=np.float32) # Global reward
        
        self.buffer_next_spatial_obs = np.zeros((capacity, episode_limit, n_agents, *spatial_obs_shape), dtype=np.float32)
        self.buffer_next_vector_obs = np.zeros((capacity, episode_limit, n_agents, vector_obs_dim), dtype=np.float32)

        self.buffer_next_global_spatial_state = np.zeros((capacity, episode_limit, *global_spatial_state_shape), dtype=np.float32)
        self.buffer_next_global_vector_state = np.zeros((capacity, episode_limit, global_vector_state_dim), dtype=np.float32)
        
        self.buffer_avail_actions = np.zeros((capacity, episode_limit, n_agents, n_actions), dtype=np.bool_)
        self.buffer_next_avail_actions = np.zeros((capacity, episode_limit, n_agents, n_actions), dtype=np.bool_)
        
        self.buffer_terminated = np.zeros((capacity, episode_limit, 1), dtype=np.bool_)
        # Padded is True if the step is beyond actual episode length
        self.buffer_padded = np.ones((capacity, episode_limit, 1), dtype=np.bool_)
        self.buffer_actual_episode_len = np.zeros((capacity,), dtype=np.int32)


        self.current_size = 0
        self.current_idx = 0

    def add_episode_data(self, episode_transitions):
        """
        Thêm một episode hoàn chỉnh vào buffer.
        episode_transitions: list các dictionary, mỗi dict chứa một transition.
        Keys: "so", "vo", "gs", "gv", "u", "r", "so_next", "vo_next",
              "gs_next", "gv_next", "avail_u", "avail_u_next", "terminated"
        """
        episode_len = len(episode_transitions)
        if episode_len == 0 or episode_len > self.episode_limit:
            print(f"Warning: Episode length {episode_len} is 0 or exceeds limit {self.episode_limit}. Skipping.")
            return

        idx = self.current_idx
        self.buffer_actual_episode_len[idx] = episode_len

        # Reset padding for this episode slot first
        self.buffer_padded[idx] = True 

        for t_idx, trans in enumerate(episode_transitions):
            self.buffer_spatial_obs[idx, t_idx] = trans["so"]
            self.buffer_vector_obs[idx, t_idx] = trans["vo"]
            self.buffer_global_spatial_state[idx, t_idx] = trans["gs"]
            self.buffer_global_vector_state[idx, t_idx] = trans["gv"]
            self.buffer_actions[idx, t_idx] = trans["u"].reshape(self.n_agents, 1) # Ensure shape (N,1)
            self.buffer_rewards[idx, t_idx] = trans["r"]
            self.buffer_next_spatial_obs[idx, t_idx] = trans["so_next"]
            self.buffer_next_vector_obs[idx, t_idx] = trans["vo_next"]
            self.buffer_next_global_spatial_state[idx, t_idx] = trans["gs_next"]
            self.buffer_next_global_vector_state[idx, t_idx] = trans["gv_next"]
            self.buffer_avail_actions[idx, t_idx] = trans["avail_u"]
            self.buffer_next_avail_actions[idx, t_idx] = trans["avail_u_next"]
            self.buffer_terminated[idx, t_idx] = trans["terminated"]
            self.buffer_padded[idx, t_idx] = False # Mark as not padded

        self.current_idx = (self.current_idx + 1) % self.capacity
        if self.current_size < self.capacity:
            self.current_size += 1

    def sample(self):
        if self.current_size < self.args.min_buffer_size_to_train or self.current_size < self.args.batch_size:
            raise ValueError(f"Replay buffer has {self.current_size} episodes, less than min buffer size to train {self.args.min_buffer_size_to_train} or batch size {self.args.batch_size}. Cannot sample.")


        indices = np.random.choice(self.current_size, self.args.batch_size, replace=False)
        device = torch.device("cuda" if self.args.use_cuda else "cpu")

        batch = {
            'so': torch.tensor(self.buffer_spatial_obs[indices], dtype=torch.float32, device=device),
            'vo': torch.tensor(self.buffer_vector_obs[indices], dtype=torch.float32, device=device),
            'gs': torch.tensor(self.buffer_global_spatial_state[indices], dtype=torch.float32, device=device),
            'gv': torch.tensor(self.buffer_global_vector_state[indices], dtype=torch.float32, device=device),
            'u': torch.tensor(self.buffer_actions[indices], dtype=torch.long, device=device),
            'r': torch.tensor(self.buffer_rewards[indices], dtype=torch.float32, device=device),
            'so_next': torch.tensor(self.buffer_next_spatial_obs[indices], dtype=torch.float32, device=device),
            'vo_next': torch.tensor(self.buffer_next_vector_obs[indices], dtype=torch.float32, device=device),
            'gs_next': torch.tensor(self.buffer_next_global_spatial_state[indices], dtype=torch.float32, device=device),
            'gv_next': torch.tensor(self.buffer_next_global_vector_state[indices], dtype=torch.float32, device=device),
            'avail_u': torch.tensor(self.buffer_avail_actions[indices], dtype=torch.float32, device=device),
            'avail_u_next': torch.tensor(self.buffer_next_avail_actions[indices], dtype=torch.float32, device=device),
            'terminated': torch.tensor(self.buffer_terminated[indices], dtype=torch.float32, device=device),
            'padded': torch.tensor(self.buffer_padded[indices], dtype=torch.float32, device=device)
        }
        return batch

    def can_sample(self):
        return self.current_size >= self.args.min_buffer_size_to_train and self.current_size >= self.args.batch_size

    def __len__(self):
        return self.current_size
