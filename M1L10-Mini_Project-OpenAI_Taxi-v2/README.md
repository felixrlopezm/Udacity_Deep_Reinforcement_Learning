# My solution to the Taxi Problem

### Getting Started

You can find the description of the environment in subsection 3.1 of [this paper](https://arxiv.org/pdf/cs/9905014.pdf).  You can verify that the description in the paper matches the OpenAI Gym environment by peeking at the code [here](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).


### Instructions

The repository contains three files:
- `agent.py`: This is where I developed my reinforcement learning agent here.  This is the only file that I modified. The other two are originals from Udacity.
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of your agent.

When you run `main.py`, the agent that you specify in `agent.py` interacts with the environment for 20,000 episodes.  The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.
- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive.  So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`.  This is the final score that you should use when determining how well your agent performed in the task.


Once you have modified the function, you need only run `python main.py` to test your new agent.

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 9.7 over 100 consecutive trials.  

### My approach and score
I have implemented both SARSA and Q-learning (aka SARSAMAX) methods; when one is executed, the other must be annotated (#).

Both methods works quite similar and, with hyperparameters indicated in the code, lead to a best_avg_reward > 9.1, as required by UDACITY Mini Project target but below the score (9.7) OpenAI fixes to consider the task solved.
