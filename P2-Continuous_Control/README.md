[//]: # (Image References)

[image1]: ./aux_items/Trained_agent.gif "Trained Agent"
[image2]: ./aux_items/Random_agent.gif "Random Agent"

# Project 2: Continuous Control

## The project

In this notebook, you can find an implementation for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), named CONTINUOUS CONTROL.

The project is about designing, training and evaluating a **smart agent** that commands a double-jointed arm (REACHER) so that its hand can reach and follow a moving target for as many time steps as possible.

The agent interacts with the so-called REACHER environment according to the following dynamics:
+ **Reward policy**. The agent receives a reward of +0.1 at each time step that the hand of the double-jointed arm is within the target location.
+ **The state space** that the agent observes has 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. State space is continuous.
+ **The action space** of the agent is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1. Action space is continous.

At the heart of the smart agent there is a deep reinforcement learning algorithm, which has been conveniently designed and trained so that the agent is able to properly command the double-joint arm to keep its arm within the target location for as many time steps as possible.

The agent's task is episodic. In order to solve the environment, it must get an average score of +30 over 100 consecutive episodes.

The algorithm herein implemented clearly improves that minimum threshold.

Trained Agent (average score +37)   |  Random Agent
:----------------------------------:|:-------------------------:
![Trained Agent][image1]            | ![Random Agent][image2]


## About the REACHER environment

The REACHER is a Unity ML-Agents environment. Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Visit [Unity ML-Agents toolkit repository in Github](https://github.com/Unity-Technologies/ml-agents) for reading more about Unity ML-Agents.

There are two versions of the REACHER environment. The first version contains a single agent. The second version contains 20 identical agents, each with its own copy of the environment. The second version is useful to distribute the task of gathering experience and is the one that has been employed in this implementation.

The REACHER environment is not included in this Github repository. Therefore, download it first if you want to run the code herein provided. To that end, follow the instructions in the "Getting Started" section.


## Getting Started
1. Clone this Github repository in your working folder.


2. Install the Unity ML-Agents Reacher environment according to the following instructions:
   + For single agent version, download the environment from one of the next links according to your operating system: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip), [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip), [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip), [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
   + For 20-agent version, download the environment from one of the next links according to your operating system: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip), [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip), [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip), [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
   
    (_for Windows users_: check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system).
   + Place the file at the root or at any folder or your choice and then, unzip the file.
   + State the path to the location of the environment file when ask to do so in the the Jupyter Notebook that runs the code.


3. Note that this project was created with Python 3.6 and with all the dependencies listed in the `requirements.txt` file. Therefore, check that your working virtual environment has, at least, that Python version and that all the dependencies are installed. Otherwise, installed them using the following command: `pip install -r requirements.txt`.

   I strongly recommend you to always create a separate virtual environment for each project. This will isolate the dependencies for one project from each other, as well as isolate them from globally installed dependencies, reducing the chance for conflict.

   If you use [Conda](https://docs.conda.io/en/latest/), run this command to list all the packages installed in your working virtual environment: `conda list`. [Check here](https://www.activestate.com/resources/quick-reads/how-to-manage-python-dependencies-with-conda/) for more details about how to manage Python dependencies in a virtual environment with Conda.  



## How to run the code

Just open the Jupiter Notebook file `My-Continuous_Control-ddpg-20a.ipynb` and follow the instructions there to run the code all by yourself!!!

Let me know if you find any bug or malfunction.

And thanks for being here!!! 😀
