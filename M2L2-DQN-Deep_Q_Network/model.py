import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_nodes=32, fc2_nodes=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Linear layer (state_size --> fc1_nodes)
        self.fc1 = nn.Linear(state_size, fc1_nodes)
        # Linear layer (fc1_nodes --> fc2_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        # Linear layer (fc2_nodes --> action_size)
        self.fc3 = nn.Linear(fc2_nodes, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))      # linear layer + reLu activation
        x = F.relu(self.fc2(x))          # linear layer + reLu activation
        x = self.fc3(x)                  # linear layer
        
        return x
