# Deep Neural Network architectures for function approximators
#

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP architecture for the state-action value function of the DQN algorithm
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_nodes=64, fc2_nodes=64):
        """Initialize parameters and define model elements.
        Params
        ======
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            seed (int): Random seed
            fc1_nodes: nodes of the first hidden layer
            fc2_nodes: nodes of the second hiddend layer
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
        """Assembling model elements for fordward pass definition.
        """
        x = F.relu(self.fc1(state))      # linear layer + reLu activation
        x = F.relu(self.fc2(x))          # linear layer + reLu activation
        x = self.fc3(x)                  # linear layer
        
        return x                         # return state-action value
