# Project 3: Competition and Collaboration
# Udacity Deep Reinforcement Learning nanodegree
#
#
# Smart Agents built with Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm
#
# Smart agents' code based on that described in the MADDPG paper.
#
#
# MADDPG paper: LOWE, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive
#               environments". arXiv preprint arXiv:1706.02275 (2017).
#
# Félix Ramón López Martínez, January 2022
#


# Import required libraries
import numpy as np
import torch

from ddpg_agent import DDPGAgent, ReplayBuffer


# AGENT class
class MADDGP_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, hyperparameters):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        BUFFER_SIZE = hyperparameters['BUFFER_SIZE']
        BATCH_SIZE = hyperparameters['BATCH_SIZE']
        self.UPDATE_EVERY = hyperparameters['UPDATE_EVERY']
        self.N_UPDATES = hyperparameters['N_UPDATES']
        RANDOM_SEED = hyperparameters['RANDOM_SEED']

        self.num_agents = num_agents
        self.action_size = action_size

        self.memory = ReplayBuffer(num_agents, action_size, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)

        self.ddpg_agents = [DDPGAgent(num_agents, state_size, action_size, hyperparameters) for _ in range(num_agents)]

        self.step_counter = 1

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience of all the agents: numpy entity (num_agents, value)
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn with UPDATE_EVERY approach
        if self.step_counter % self.UPDATE_EVERY == 0:
            for _ in range(self.N_UPDATES):
                for agent in self.ddpg_agents:
                    agent.step(self.memory)

        # Learn without UPDATE_EVERY approach
        #for agent in self.ddpg_agents:
            #agent.step(self.memory)

        self.step_counter += 1

    def act(self, states, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = np.zeros([self.num_agents, self.action_size])
        for i, agent in enumerate(self.ddpg_agents):
            actions[i,:] = agent.act(states[i], i_episode, add_noise)
        return actions

    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
