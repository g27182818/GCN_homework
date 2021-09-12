# GCN_homework

In this homework we will explore the basics of the Pytorch-Geometric Package through an initial tutorial and implement different a state of the art Graph Neural Networks to solve a node classification task in the OGBN-Proteins dataset.

# Part 0: Installation

The [Installation file](deepgcn_env_install.txt) contains all needed comands with respect to the installation of required packages and the activation of the respective environment in anaconda. It is recommended to run all lines sequentially.

# Part 1: Pytorch Geometric Tutorial (0.5 points)

The `PyTorch Geometric Tutorial` folder contains the [tutorial file](https://github.com/g27182818/GCN_homework/blob/7e4b826915b055442b79ebf4c981dc50ae44a111/PyTorch%20Geometric%20Tutorial/TorchGeometric_tutorial.py) which is a step by step guide to the construction of a basig graph neural network using the [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club) dataset. This tutorial is an adaptation from the [original](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8?usp=sharing#scrollTo=ci-LpZWhRJoI) by [MATTHIAS FEY](https://rusty1s.github.io/#/). Please remit to the folder for detailed instructions. To obtain the points, you must append to your report the final figure of the tutorial and answer the following questions:

1. What is the average node degree in the [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club) network?
2. From the final 2D embedings, is the network producing a perfect classifier for all nodes in the graph?

# Part 2: Dataset downloading and exploration (0.5 points)

In this part you will be downloading and exploring the [OGBN-Proteins dataset](https://ogb.stanford.edu/docs/nodeprop/). Following the instructions from the official website produce a script (And name it `ogbn_proteins_explore.py`) that will download and explore the dataset (Use some of the functions learned in the tutorial). To get the points you have to append a table in your report with descriptive features about the dataset and answer the following questions:

1. What is the task for this dataset?
2. How are the graph features organized?
3. Which is the evaluation metric of this dataset and how is it analitically calculated?

# Part 3: Naive approximation (1 point)

# Part 4: Plain State of the art implementation (1 point)

# Part 5: Residual implementation (1 point)

# Part 6: Hyperparameter experimentation (1 point)
