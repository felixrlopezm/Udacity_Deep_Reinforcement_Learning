[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135612-cbff24aa-7d12-11e8-9b6c-2b41e64b3bb0.gif "Trained Agent"

# Deep Q-Network (DQN)

### Instructions

In this exercise, I implemented a Deep Q-Learning to solve OpenAI Gym's LunarLander-v2 environment.

The folder contains three files:
+	Deep_Q_Network.ipynb: a Jupyther notebook managing the whole process.
+	Dqn_agent.py: agent class definition; replay buffer definition; the task here was to finish the learn method; I followed the same implementation than Udacity provided solution
+	model.py: the task here was to define the neural network architecture that maps states to action values (Q-network); I implemented a Q-Network built with a fully connected neural network: three linear layers; hidden layers with 32 nodes (Udacity solution was 64 but I prefered a lighter version just to check the agent was able to learn; reLu activation function applied to the first two layers

### Learning phase
The agent was trained for 2000 episodes according to the following log:
[image]("/Agent-DQN-32nodes-2k_episodes.png")

### Results

![Trained Agent][image1]

### Resources

- [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
