[//]: # (Image References)

[image1]: ./aux_items/Trained_agent.gif "Trained Agent"
[image2]: ./aux_items/Random_agent.gif "Random Agent"

# Project 1: Navigation

## The project

In this notebook, you can find an implementation for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), called NAVIGATION.

The project is about designing, training and evaluating a **smart agent** whose goal is to solve a Banana Collector environment. This environment is a square world in which there are yellow and blue bananas spread across its surface.

The agent interacts with the environment according to the following dynamics:
+ **Reward policy**. The agent receives a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana. 
+ **The state space** that the agent observes has 37 dimensions and contains its velocity, along with ray-based perception of objects around the agent's forward direction.
+ **The action space** of the agent is made up of four discrete actions corresponding to:
  + 0 - walk forward;
  + 1 - walk backward;
  + 2 - turn left;
  + 3 - turn right.

At the heart of the smart agent there is a deep reinforcement learning algorithm, which has been conveniently designed and trained so that the agent is able to properly navigate in the environment in order to collect yellow bananas and avoid the blue ones.

The agent faces an episodic task. **To solve the environment**, the agent must get an average score of +13 over 100 consecutive episodes.

The algorithm herein implemented clearly improves that minimum threshold and definitely works much better than a random agent. 

Trained Agent (average score +16)   |  Random Agent
:----------------------------------:|:-------------------------:
![Trained Agent][image1]            | ![Random Agent][image2] 


## About the Banana Collector environment

The Banana collector environment is a Udacity-dedicated version of the Unity ML-Agents environment "Food Collector". Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents. Visit [Unity ML-Agents toolkit repository in Github](https://github.com/Unity-Technologies/ml-agents) for reading more about Unity ML-Agents.

This environment is not included in this Github repository. Therefore, download it first if you want to run the code herein provided. To that end, follow the instructions in the "Getting Started" section.


## Getting Started
1. Clone this Github repository in your working folder.


2. Install the Banana Collector environment according to the following instructions:
   + Download the environment from one of the next links according to your operating system: [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip), [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip), [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip), [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    (_for Windows users_: check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system).
   + Place the file at the root or at any folder or your choice and then, unzip the file.
   + State the path to the location of the environment file when ask to do so in the the Jupyter Notebook that runs the code.
   

3. Note that this project was created with Python 3.6 and with all the dependencies listed in the `requirements.txt` file. Therefore, check that your working virtual environment has, at least, that Python version and that all the dependencies are installed. Otherwise, installed them using the following command: `pip install -r requirements.txt`.

   I strongly recommend you to always create a separate virtual environment for each project. This will isolate the dependencies for one project from each other, as well as isolate them from globally installed dependencies, reducing the chance for conflict.
    
   If you use [Conda](https://docs.conda.io/en/latest/), run this command to list all the packages installed in your working virtual environment: `conda list`. [Check here](https://www.activestate.com/resources/quick-reads/how-to-manage-python-dependencies-with-conda/) for more details about how to manage Python dependencies in a virtual environment with Conda.  



## How to run the code
 
Just open the Jupiter Notebook file `MY-Navigation.ipynb` and follow its instructions to run the code all by yourself!!!

Let me know if you find any bug or malfunction.

And thanks for being here!!! 😀