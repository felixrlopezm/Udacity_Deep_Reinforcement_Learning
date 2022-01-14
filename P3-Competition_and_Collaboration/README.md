[//]: # (Image References)

[image1]: ./aux_items/Trained_agent.gif "Trained Agent"
[image2]: ./aux_items/Random_agent.gif "Random Agent"

# Project 3: Collaboration and Competition. Tennis

## The project

In this notebook, you can find an implementation for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), named COLLABORATION & COMPETITION.

The project is about designing, training and evaluating a **pair of smart agents** that play TENNIS against the other. More specifically, each agent controls a racket and has to bounce a ball over a net and not let the ball drop. **The goal of each agent is to keep the ball in play when comes from the other side of the court**.

The agent interacts with the so-called REACHER environment according to the following dynamics:
+ **Reward policy**. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
+ **The state space** that every agent observes has 8 variables corresponding to the position and velocity of the ball and racket. State space is continuous. Moreover, every agent receives three stacked vector observations per time step, so that the agent is able to capture the sense of movement.
+ **The action space** of every agent is a vector with two actions, corresponding to movement toward (or away from) the net and jumping. Every entry in the action vector must be a number between -1 and 1. Action space is continuous.

At the heart of every smart agent there is a deep reinforcement learning algorithm, which has been conveniently designed and trained so that the agent is able to properly command the racket and, therefore, keep the ball in play for as many time steps as possible.

The task is episodic. In order to solve the environment, any of the agents must get an average score of at least +0.5 over 100 consecutive episodes.

The algorithm herein implemented clearly improves that minimum threshold.

Trained Agent (average score +0.7)   |  Random Agent
:----------------------------------:|:-------------------------:
![Trained Agent][image1]            | ![Random Agent][image2]


## About the TENNIS environment

The TENNIS environment is a Udacity-dedicated version of the Unity ML-Agents TENNIS environment. Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Visit [Unity ML-Agents toolkit repository in Github](https://github.com/Unity-Technologies/ml-agents) for reading more about Unity ML-Agents.

The TENNIS environment is not included in this Github repository. Therefore, download it first if you want to run the code herein provided. To that end, follow the instructions in the "Getting Started" section.


## Getting Started
1. Clone this Github repository in your working folder.


2. Install the TENNIS environment according to the following instructions:
   + Download the environment from one of the next links according to your operating system: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip), [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip), [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip), [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_for Windows users_: check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system).
   + Place the file at the root or at any folder of your choice and then, unzip the file there.
   + When ask to do so in the the Jupyter Notebook that runs the code, set the path to the location of the environment file .


3. Note that this project was created with Python 3.6 and with all the dependencies listed in the `requirements.txt` file. Therefore, check that your working virtual environment has, at least, that Python version and that all the dependencies are installed. Otherwise, installed them using the following command: `pip install -r requirements.txt`.

   I strongly recommend you to always create a separate virtual environment for each project. This will isolate the dependencies for one project from each other, as well as isolate them from globally installed dependencies, reducing the chance for conflict.

   If you use [Conda](https://docs.conda.io/en/latest/), run this command to list all the packages installed in your working virtual environment: `conda list`. [Check here](https://www.activestate.com/resources/quick-reads/how-to-manage-python-dependencies-with-conda/) for more details about how to manage Python dependencies in a virtual environment with Conda.  



## How to run the code

Just open the Jupiter Notebook file `My-Tennis-maddpg.ipynb` and follow the instructions there to run the code all by yourself!!!

Let me know if you find any bug or malfunction.

And thanks for being here!!! 😀
