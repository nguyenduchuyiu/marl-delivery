# networks.py
import torch    
import torch.nn as nn
import torch.nn.functional as F
 

class ActorNetwork(nn.Module):
    def __init__(self, spatial_obs_shape, vector_obs_dim, action_dim=15,
                 cnn_channels_out=64, mlp_hidden_dim=256, combined_hidden_dim=256):
        """
        Actor Network that processes both spatial and vector observations.

        Args:
            spatial_obs_shape (tuple): Shape of the spatial observation (C_map, H, W).
                                       e.g., (6, n_rows, n_cols) from convert_observation.
            vector_obs_dim (int): Dimension of the vector observation.
                                  e.g., output size of generate_vector_features.
            action_dim (int): Total number of discrete actions.
            cnn_channels_out (int): Number of output channels from the last CNN layer.
            mlp_hidden_dim (int): Hidden dimension for the vector processing MLP.
            combined_hidden_dim (int): Hidden dimension for the combined MLP.
        """
        super(ActorNetwork, self).__init__()
        self.spatial_channels_in = spatial_obs_shape[0]
        self.map_h = spatial_obs_shape[1]
        self.map_w = spatial_obs_shape[2]

        # --- CNN Branch for Spatial Observations ---
        self.conv1 = nn.Conv2d(self.spatial_channels_in, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, cnn_channels_out, kernel_size=3, stride=1, padding=1) # Last conv layer
        self.bn3 = nn.BatchNorm2d(cnn_channels_out)
        
        # Adaptive pooling to get a fixed size output from CNN, regardless of map_h, map_w (within reason)
        # This avoids calculating flattened_size manually if map dimensions might vary slightly
        # or if you want more flexibility. Output size of (e.g., 4x4) from pooling.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.cnn_flattened_dim = cnn_channels_out * 4 * 4 # Output from adaptive_pool

        # --- MLP Branch for Vector Observations ---
        self.vector_fc1 = nn.Linear(vector_obs_dim, mlp_hidden_dim)
        self.vector_fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2) # Reduce dim slightly

        # --- Combined MLP ---
        # Input to this MLP is the concatenation of CNN output and Vector MLP output
        combined_input_dim = self.cnn_flattened_dim + (mlp_hidden_dim // 2)
        self.combined_fc1 = nn.Linear(combined_input_dim, combined_hidden_dim)
        self.actor_head = nn.Linear(combined_hidden_dim, action_dim)

    def forward(self, spatial_obs, vector_obs):
        """
        Args:
            spatial_obs (torch.Tensor): (batch_size, C_map, H, W)
            vector_obs (torch.Tensor): (batch_size, vector_obs_dim)
        Returns:
            action_logits (torch.Tensor): (batch_size, action_dim)
        """
        # CNN path
        x_spatial = F.relu(self.bn1(self.conv1(spatial_obs)))
        x_spatial = F.relu(self.bn2(self.conv2(x_spatial)))
        x_spatial = F.relu(self.bn3(self.conv3(x_spatial)))
        x_spatial = self.adaptive_pool(x_spatial)
        x_spatial_flat = x_spatial.reshape(x_spatial.size(0), -1) # Flatten

        # Vector MLP path
        x_vector = F.relu(self.vector_fc1(vector_obs))
        x_vector_processed = F.relu(self.vector_fc2(x_vector))

        # Concatenate processed features
        combined_features = torch.cat((x_spatial_flat, x_vector_processed), dim=1)

        # Combined MLP path
        x_combined = F.relu(self.combined_fc1(combined_features))
        action_logits = self.actor_head(x_combined)

        return action_logits

class CriticNetwork(nn.Module):
    def __init__(self, global_spatial_state_shape, global_vector_state_dim,
                 cnn_channels_out=64, mlp_hidden_dim=256, combined_hidden_dim=256):
        """
        Critic Network that processes both global spatial and global vector states.

        Args:
            global_spatial_state_shape (tuple): Shape of the global spatial state (C_global_map, H, W).
                                                e.g., (4, n_rows, n_cols) from get_global_state_for_critic.
            global_vector_state_dim (int): Dimension of the global vector state.
            cnn_channels_out (int): Number of output channels from the last CNN layer.
            mlp_hidden_dim (int): Hidden dimension for the vector processing MLP.
            combined_hidden_dim (int): Hidden dimension for the combined MLP.
        """
        super(CriticNetwork, self).__init__()
        self.global_spatial_channels_in = global_spatial_state_shape[0]
        self.map_h = global_spatial_state_shape[1]
        self.map_w = global_spatial_state_shape[2]

        # --- CNN Branch for Global Spatial State ---
        self.conv1 = nn.Conv2d(self.global_spatial_channels_in, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, cnn_channels_out, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(cnn_channels_out)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4)) # Fixed output size
        self.cnn_flattened_dim = cnn_channels_out * 4 * 4

        # --- MLP Branch for Global Vector State ---
        self.vector_fc1 = nn.Linear(global_vector_state_dim, mlp_hidden_dim)
        self.vector_fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2)

        # --- Combined MLP ---
        combined_input_dim = self.cnn_flattened_dim + (mlp_hidden_dim // 2)
        self.combined_fc1 = nn.Linear(combined_input_dim, combined_hidden_dim)
        self.critic_head = nn.Linear(combined_hidden_dim, 1) # Outputs a single state value

    def forward(self, global_spatial_state, global_vector_state):
        """
        Args:
            global_spatial_state (torch.Tensor): (batch_size, C_global_map, H, W)
            global_vector_state (torch.Tensor): (batch_size, global_vector_state_dim)
        Returns:
            value (torch.Tensor): (batch_size, 1)
        """
        # CNN path
        x_spatial = F.relu(self.bn1(self.conv1(global_spatial_state)))
        x_spatial = F.relu(self.bn2(self.conv2(x_spatial)))
        x_spatial = F.relu(self.bn3(self.conv3(x_spatial)))
        x_spatial = self.adaptive_pool(x_spatial)
        x_spatial_flat = x_spatial.reshape(x_spatial.size(0), -1)

        # Vector MLP path
        x_vector = F.relu(self.vector_fc1(global_vector_state))
        x_vector_processed = F.relu(self.vector_fc2(x_vector))

        # Concatenate processed features
        combined_features = torch.cat((x_spatial_flat, x_vector_processed), dim=1)

        # Combined MLP path
        x_combined = F.relu(self.combined_fc1(combined_features))
        value = self.critic_head(x_combined)

        return value