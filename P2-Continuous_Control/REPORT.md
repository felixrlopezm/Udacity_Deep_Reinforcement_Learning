[//]: # (Image References)

[image1]: ./aux_items/Learning_Process-1.png "Learning Process: general view"
[image2]: ./aux_items/Learning_Process-2.png "Learnig Process: detail"

# Technical report on Project 2: Continuous Control

## Introduction

This report introduces the details about how the smart agent has been designed and trained so that it can solve the task posed by the CONTINUOUS CONTROL project, namely: to command a double-joint arm (REACHER) in the REACHER environment so that its hand can reach and follow a moving target for as many time steps as possible

## Learning algorithm
The smart agent is taught to optimally command the double-joint arm following a Deep Reinforcement Learning approach.

Whereas Reinforcement Learning is a computational framework that addresses the problem of teaching an agent to select actions in a certain environment, with which it interacts, Deep Reinforcement Learning can deal with the same problem in unknown, dynamic, complex and even stochastic environments. And this is the case of the REACHER environment: it is a continuous space that the agent observes through a large number of sensory inputs (33) and on which it has no prior knowledge.

The smart agent is designed on the top of the Deep Deterministic Policy Gradient (DDPG) algorithm, which was first introduced in 2019 in [this paper](https://arxiv.org/pdf/1509.02971.pdf) by a team of researchers of Google Deepmind. The algorithm is, basically, a clever modification of the [DQN algorithm](http://files.davidqiu.com//research/nature14236.pdf) in order to overcome its limitation with regard to the application to continuous action spaces, as it is the case in the REACHER environment.

Likewise Actor-Critic algorithms, DDPG uses two deep neural networks: one, the so-called actor, acts as a function approximation of the policy that maps state to action; the other, the critic, is the function approximation of the action-value function. The actor network is used to approximate the optimal policy deterministically; that means, it always outputs the best believed action for any given state. Therefore, the actor is basically learning the argmaxaQ(s, a), which is the best action. On the other hand, the critic learns to evaluate the optimal action-value function by using the actors best believed action.

Both networks interacts this way. The actor network is used to evaluate the best believed next action, which is then used to calculate a new target value for the action-value function, against which the critic network is optimized by minimizing its loss (as in the DQN algorithm). On its part, the critic network is optimized by means of policy gradient that maximizes the action-value function, calculated with the critic network, for the best believed action provided by the actor network.

Likewise DQN algorithm, DDPG employs **Fixed Targets** to avoid harmful correlations between the target and the parameters that change during the optimization (update a guess with a guess). This is, it uses two copies of the network weights for each network: a regular for the actor and a regular for the critic; and a target for the actor and a target for the critic. Then, the target networks are updated with the weight of the regular networks using a **soft update strategy**. It consists of slowly blending the regular network weights with the target network weights. For instance, for a blending factor (tau) of 0.01, the target network of both actor and critic networks will be made 99.99% of the target network weights and only 0.01% of the regular network weights.

Moreover fixed targets and soft update strategy, the DDPG paper also employs the following techniques for addressing other specific challenges:
+ As in DQN, **Replay Buffer** is used to break the correlation effects between consecutive experience tuples.
+ **Batch normalization** is used to reconcile different physical units in the vector observation of the state.
+ **Noise** is added to the actor policy in order to push for exploration during the learning process. Ornstein-Uhlenbeck noise is used by the authors of the DDPG paper.

All these techniques have been considered during the implementation of the smart agent algorithm herein described. Furthermore, all of them are found effective to solve the REACH environment except for batch normalization, which makes no impact on the algorithm performance, perhaps because the environment already returns a normalized state vector.

The smart agent is codified in Python-PyTorch code in the file `ddpg_agent.py`.

## Architecture of the neural network used as function approximation for Actor and Critic functions
At the heart of the Actor and Critic functions there are two deep neural network that act as function approximation of the policy function and the action-value function respectively. For the actor, the environment state is passed in and the function approximation produces an action vector. For the critic, the environment state and the action vector (output from the actor) are passed in and the function approximation produces the action-value for that state-action pair.

Both function approximations are represented by similar feedforward multilayer perceptron (MLP) neural network with the following features:
+ input layer: 33 nodes (one for every dimension of the state vector)
+ two hidden layers with 128 nodes each and a ReLu activation function applied at the output of each layer
+ output layer: 4 nodes (one for every dimension of the action space) and tanh activation function for the actor; and 1 node (the action-value) without any activation function for the critic.
+ for the actor network, the state vector is introduced in the first layer of the network
+ for the critic network, the state vector is introduced in the first layer of the network and then its output is concatenated with the action vector before being input in the second layer of the network

The architectures of the actor and critic function approximations are codified in Python-PyTorch code in the file `actor_critic_models.py`.

## Chosen Hyperparameters
The final values of the considered hyperparameters are listed next.

+ Function approximator architecture for both actor and critic functions:
  + Nodes at hidden layer1 --> fc1 = 128
  + Nodes at hidden layer2 --> fc2 = 128

+ Learning process
  + Replay buffer size = 100000
  + Minibatch size = 128
  + Discount factor (gamma) = 0.93
  + Target parameters soft update factor (TAU) = 0.001
  + Actor Learning rate = 0.0005
  + Critic Learning rate = 0.0001
  + Critic network L2 weight decay = 0
  + Update the actor-critic networks every 20 time steps
  + Number of learning passes at every update = 10
  + Maximum time-steps per episode = until the end of the episode (done = True)

+ Ornstein-Uhlenbeck noise parameters:
  + Mu = 0
  + Sigma = 0.05
  + Theta = 0.10

Those final values are the result of an heuristic approach. The starting point was the values stated in the DDGP paper. From there, the trials mainly focused on variation of: actor and critic learning rates; number of nodes of the network hidden layers (fc1, fc2); an sigma and theta noise parameters. As suggested in the benchmark implementation of this project, a strategy for dealing with the number of updates per time step was introduced to help stabilizing the learning process. Once a promising combination of hyperparameters was achieved, fine tuning of discount factor and actor/critics learning rates let find the final values.

The hyperparameter tweaking process revealed the following:
+ The hyperparamenter values stated in the DDGP paper did not work at all for solving the REACHER environment.
+ Not considering Critic network L2 weight decay seems to have no impact on the final solution
+ The REACHER environment can be solved with less nodes in the hidden layers of actor/critic neural networks than those used in the DDPG paper
+ It has not been observed that using Gradient clipping when training the critic network makes a difference in the stability of the learning process (contrary to that stated in the benchmark implementation of this project))
+ Contrary to the DDPG paper, a greater learning for the actor network than for the critic network work much better than an inverse relation between them as that shown in the DDPG paper.
+ The addition of noise to the action vector has found decisive for solving the REACHER environment. Furthermore,the stability of the learning process is quite sensitive to sigma-theta combination (parameters that define the Ornstein-Uhlenbeck noise)
+ The whole stability of the learning process is quite sensitive to small variations of the following hyperparameters: learning rates, discount factor and noise parameters. This makes the process of tuning the algorithm challenging and time costing.


## Smart agent performance

With the learning algorithm chosen for the smart agent and the hyperparameters listed in the previous section, the typical learning process of the agent is shown in the figures below (general view until 400 episodes and detailed view zoomed at 150 episodes)

Figure 1: general view              |  Figure 2: detail
:----------------------------------:|:-------------------------:
!
!![Learning Process general][image1]  | ![Learning Process detail][image2]

The plot of the learning process shows that the algorithm learns smoothly and with a progressive growth rate until closed to episode 100. From that point, the learning process enters in a stable learning plateau with an average score of some 37.5 points and a maximum value of 38.10 points at episode 205. The plot shows that the plateau behavior spans (at least) until episode 400, at which time the learning algorithm was forced to stop.

**REACHER environment can be considered solved at episode 146**, where the average score of the last 100 episodes is over 30 points.

The weights of the neural network of the action-value function approximation at the maximum-score episode are saved in the files `trained_actor-ddpg.pth` and `trained_critic-ddpg.pth` . They can be used to feed the actor and critic networks of the trained agent.


## Ideas for future work
In my opinion, the smart agent built with the DDPG algorithm and tuned with the listed hyperparameters has little room for improvement. Therefore, it is worth considering whether another type of Deep Reinforcement Learning algorithm will have better performance in terms of stability of the learning process, speed of the learning process or maximum score of the trained agent.

To that end, some alternative algorithms for continuous control tasks are:
+ [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
+ [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
+ [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://openreview.net/pdf?id=SyZipzbCb)
+ [Asynchronous Advantage Actor Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
+ Advantage Actor Critic (A2C)
