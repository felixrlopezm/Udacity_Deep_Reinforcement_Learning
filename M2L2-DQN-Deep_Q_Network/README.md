[//]: # (Image References)

[image1]: Agent-DQN-32nodes-2k_episodes.png "training log"

# Deep Q-Network (DQN)

### Instructions

In this exercise, I implemented a Deep Q-Learning to solve OpenAI Gym's LunarLander-v2 environment.

The folder contains three files:
+	Deep_Q_Network.ipynb: a Jupyther notebook managing the whole process.
+	Dqn_agent.py: agent class definition; replay buffer definition; the task here was to finish the learn method; I applied the same implementation than Udacity provided in the solution
+	model.py: the task here was to define the neural network architecture that maps states to action values (Q-network); I implemented a Q-Network built with a fully connected neural network: three linear layers; hidden layers with 32 nodes (Udacity solution was 64 but I prefered a lighter version just to check the agent was able to learn; reLu activation function applied to the first two layers

### Learning phase
The agent was trained for 2000 episodes according to the following log:

![Learnig log][image1]

### Trained agent
Trained agent is stored in the files: checkpoint-32nodes-2k_episodes-cpu.pth (cpu version); checkpoint-32nodes-2k_episodes-cuda.pth (cuda version)

To see the performance of the trained agent, load the appropriate checkpoint (cpu o cuda) in the local Q-networt with code at step 4 of the Jupyter Notebook

### Resources

- [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
