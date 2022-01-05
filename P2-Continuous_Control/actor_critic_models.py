# Project 2: continuous Control
# Udacity Deep Reinforcement Learning nanodegree
#
#
# Deep Neural Network architectures for function approximators
# of DDPG agent and critic functions
#
# Architectures based on those used in the DDPG paper and modify accordingly to solve
# the task at hand. In particular, no batch normalization used (corresponding code commented)
#
# DDPG paper: Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." 
#             arXiv preprint arXiv:1509.02971 (2015).
# 
# Félix Ramón López Martínez, January 2022
#
# This implementation is a modified version of the original  
# Alexix Cook code for the ddpg-pendulum example in
# Udacity Deep Reinforcement Learning Nanodegree


# Import required libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Weights and biases hidden layers initialization
# Uniform distribution acc. to DDPG paper
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Actor class definition
class Actor(nn.Module):
    """Actor (Policy) model to map states to actions
       Model architecture: feedforward neural network.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and define model elements.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Manual seed for repetitiveness
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)  

        # Definition of linear feedforward layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # Batch normalization (acc. to DDPG paper)
        #self.bn1 = nn.BatchNorm1d(state_size)
        #self.bn2 = nn.BatchNorm1d(fc1_units)
        #self.bn3 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):                               # acc. to DDPG paper
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)              

    def forward(self, state):
        """Network Forward pass maps state to actions"""
        #if state.dim() == 1:                           # If needed, add required 2nd dimension for batch normalitation
        #    state = torch.unsqueeze(state, 0)
        #x = self.bn1(state)
        x = F.relu(self.fc1(state))
        #x = self.bn2(x)
        x = F.relu(self.fc2(x))
        #x = self.bn3(x)
        action = F.tanh(self.fc3(x))

        return action

# Critic class definition
class Critic(nn.Module):
    """Critic (Value) Model to map states to state-values.
       Model architecture: feedforward neural network.
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Definition of linear feedforward layers
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        # Batch normalization (acc. to DDPG paper)
        #self.bn1 = nn.BatchNorm1d(state_size)
        #self.bn2 = nn.BatchNorm1d(fcs1_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Network Forward pass maps maps (state, action) pairs to state-values.
        """
        #if state.dim() == 1:                    # If needed, add required 2nd dimension for batch normalitation
        #    state = torch.unsqueeze(state, 0)
        #xs = self.bn1(state)
        xs = F.relu(self.fcs1(state))
        #xs = self.bn2(xs)
        x = torch.cat((xs, action.float()), dim=1)
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        return value
