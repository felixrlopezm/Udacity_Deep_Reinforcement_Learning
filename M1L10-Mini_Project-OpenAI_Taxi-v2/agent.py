import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon_start = 1.0
        self.i_episode = 1.0
        self.alpha = 0.04
        self.gamma = 0.9

    def epsilon_greedy_probs(self, state, epsilon):
        ''' Calculation of probabilities accordgin to a 
        epsilon greedy policy'''
        probs = np.ones(self.nA) * epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        probs[best_action] = 1 - epsilon + (epsilon / self.nA)
        
        return probs
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # Random action
        # action = np.random.choice(self.nA)
        
        # Epsilon decay
        epsilon = self.epsilon_start / self.i_episode
        # Epsilon-greedy policy/probabilities
        probs = self.epsilon_greedy_probs(state, epsilon)
        # Action selection acc. to epsilon-greedy policy
        action = np.random.choice(np.arange(self.nA), p = probs)
        
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
       # SARSA method
        next_action = self.select_action(next_state)
        Gt = reward + self.gamma * self.Q[next_state][next_action]
                                   
        # Q-learning (SARSAMAX) method
        #best_action = np.argmax(self.Q[next_state])
        #Gt = reward + self.gamma * self.Q[next_state][best_action]
            
        self.Q[state][action] += self.alpha * (Gt - self.Q[state][action])
        
        # i_episode update for calculation of epsilon decay
        self.i_episode += 1.0