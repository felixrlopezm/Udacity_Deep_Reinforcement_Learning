# Project 3: Competition and Collaboration
# Udacity Deep Reinforcement Learning nanodegree
#
#
# Smart Agents built with Multi Agent Deep Deterministic Policy Gradient (MADDPG) algorithm
#
# Smart agents' code based on that described in the DDPG paper and modify accordingly to fit
# with the MADDPG algorithm. In particular, Repplay Buffer and Leaning process (learn function)
# have been modified accordingly.
#
# DDPG paper: Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning."
#             arXiv preprint arXiv:1509.02971 (2015).
# MADDPG paper: LOWE, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive
#               environments". arXiv preprint arXiv:1706.02275 (2017).
#
#
# Félix Ramón López Martínez, January 2022
#
# This implementation is a modified version of the
# original Alexix Cook code for the ddpg-pendulum example in
# Udacity Deep Reinforcement Learning Nanodegree



# Import required libraries
import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
from actor_critic_nets import Actor, Critic

# GPU-CPU device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# AGENT class
class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, hyperparameters):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.FC1 = hyperparameters['FC1']
        self.FC2 = hyperparameters['FC2']
        self.BATCH_SIZE = hyperparameters['BATCH_SIZE']
        self.GAMMA = hyperparameters['GAMMA']
        self.TAU = hyperparameters['TAU']
        self.LR_ACTOR = hyperparameters['LR_ACTOR']
        self.LR_CRITIC = hyperparameters['LR_CRITIC']
        self.WEIGHT_DECAY = hyperparameters['WEIGHT_DECAY']
        self.MU = hyperparameters['MU']
        self.SIGMA = hyperparameters['SIGMA']
        self.THETA = hyperparameters['THETA']
        self.RANDOM_SEED = hyperparameters['RANDOM_SEED']
        self.EPS_INT = hyperparameters['EPS_INT']

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, self.RANDOM_SEED, self.FC1, self.FC2).to(device)
        self.actor_target = Actor(state_size, action_size, self.RANDOM_SEED, self.FC1, self.FC2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(num_agents, state_size, action_size, self.RANDOM_SEED, self.FC1, self.FC2).to(device)
        self.critic_target = Critic(num_agents, state_size, action_size, self.RANDOM_SEED, self.FC1, self.FC2).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, self.RANDOM_SEED, self.MU, self.THETA, self.SIGMA)

        # Noise decay setup
        self.eps_start = 1
        eps_end = 0
        self.eps = self.eps_start
        self.eps_decay = (self.eps_start - eps_end) / self.EPS_INT


    def step(self, memory):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # Save experience / reward   --> in MADDGP, the replay buffer is fed from the MADDPG agent
        #self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(memory) > self.BATCH_SIZE:
            experiences = memory.sample()
            self.learn(experiences, self.GAMMA)

    def act(self, state, i_episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        # Evaluation of action vector (numpy) with actor_local policy
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # Noise addition to action vector
        if add_noise:
            self.eps = self.eps_start - self.eps_decay * (i_episode-1)  # Update of Noise decay factor
            action += self.noise.sample() * self.eps
            #action += self.noise.sample() * self.eps

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()       # Reset noise

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_list, actions_list, rewards, next_states_list, dones = experiences

        # Concatente all items in the lists and turn into a torch entity
        states = torch.cat(states_list, dim=1).to(device)
        actions = torch.cat(actions_list, dim=1).to(device)
        next_states = torch.cat(next_states_list, dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Compute predicted next-state actions from target (ant turn into torch)
        next_actions_list = [self.actor_target(next_state) for next_state in next_states_list]
        next_actions = torch.cat(next_actions_list, dim = 1).to(device)

        # Compute Q target for next_states-next_actions
        Q_target_next = self.critic_target.forward(next_states, next_actions)

        # Compute Q targets (y_i)
        Q_target = rewards + (gamma * Q_target_next * (1 - dones))

        # Compute Q expected form local critic
        Q_expected = self.critic_local.forward(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)       # Gradient clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute new actions (and turn into torch entity)
        actions_pred_list = [self.actor_local(state) for state in states_list]
        actions_pred = torch.cat(actions_pred_list, dim=1).to(device)

        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0-tau) * target_param.data)


# Ornstein-Uhlenbeck noise class
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])  # lower performance
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


# Replay Bufffer class
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agents, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agents = num_agents
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory.
           Note that states, actions and next_states are provided
           as a sorted list with the values of all the agents involved
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states_list = [torch.from_numpy(np.vstack([e.states[i] for e in experiences if e is not None])).float().to(device) for i in range(self.num_agents)]
        actions_list = [torch.from_numpy(np.vstack([e.actions[i] for e in experiences if e is not None])).float().to(device) for i in range(self.num_agents)]
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states_list = [torch.from_numpy(np.vstack([e.next_states[i] for e in experiences if e is not None])).float().to(device) for i in range(self.num_agents)]
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states_list, actions_list, rewards, next_states_list, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
