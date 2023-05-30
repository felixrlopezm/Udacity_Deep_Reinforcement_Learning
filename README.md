[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Udacity Deep Reinforcement Learning

In this repository, I stored my implementations to the examples, micro projects and projects of the Udaciy Deep Reinforcement Learning Nanodegree program.

In general, I wrote my code using the starter code provided by Udacity as the take-off point and then modified it or added my whole own code where necessary. I also included additional remarks and comments to those already provided by Udacity for easing the understanding of the resulting code.

I started the nanodegree in August 2021 and finished in January 2022.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. To install the base Gym library, use pip install gym. This basic instalation includes the **classic control** environment group and the **box2d** environment group.

   To install the **Atari** environment group, use conda and run: ```conda install -c conda-forge gym-atari```
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

