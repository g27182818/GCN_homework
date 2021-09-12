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

The `Naive Aproximation` folder contains a [first approach file](https://github.com/g27182818/GCN_homework/blob/48352d34cf625421610f1a7d0681354c0fd8e7c0/Naive%20approximation/GCN_ogbn_naive.py). This file constructs a GNN based on the one proposed on the tutorial.

Your first task is to add parts of your previous script to load the dataset correctly. Once you have done this, you will be able to train and evaluate the network (A final figure with training loss and metrics in train and val will be produced).

It is important to note that this network **does not work properly**. Hence, your task is to understand why this happens. To obtain the points you must:

1. Append the final plot produced by your code in your report.
2. Add a breakpoint in the final embedings of the nodes (epoch 100) and seek for possible problems in terms of what was explained on class.
3. Answer: why doesn't this network work properly? You can use histograms, simple statistics or other kind of data to support your answer.
 

# Part 4: Plain State of the art implementation (1 point)

In this part of the homework your task is to implement some state of the art GNN's to solve the problem initially seen in part 3 (this problem is actually due to architecture faliures which are corrected in this new implementation). You will also dive inside the code and implement a new convolutional operator that was not originally published with this method. You will only be modifying the [model.py](https://github.com/g27182818/GCN_homework/blob/5a4d540a21954f4146d373b96fff677906c3b773/deep_gcns/model.py) file inside the `deep_gcns` folder. Please just modify the parts of the code that explicitlly ask to do so, otherwise the whole method could stop working. To obtain the points you have to:

1. Read the `model.py`, `main.py`, `dataset.py` and `args.py` files. You do not have to understand everything but you need to have a general idea of what is happenning inside the code. Answer: Is this method evaluating the whole training set in every iteration as our previous code was doing? How are they computing node features?
2. Run the `PlainGCN` configuration for 100 epochs and obtain a baseline for your experiments. Append to your report the highest metrics obtained in train, validation and test. For reference, it should take **one and a half** hours to run the whole experiment.
3. Change the `model.py` file to implement a GAT with one head (you shall import convolutional operators from `deep_gcns/gcn_lib/sparse/torch_vertex.py`). Obtain performance metrics and discuss according to what was explained in class. **Bonus (1 point) implement a multi-head GAT (from 2 to 8 heads). This might require to change parts of the code not previously specified.**    

# Part 5: Residual implementation (1 point)

Change the `model.py` file to implement a residual GNN. Obtain performance metrics and discuss according to what was explained in class. Answer: Which of the 3 implementations (`PlainGCN`, `GATPlainGCN`, `ResGCN`) resulted in better results. Why do you think this is the case?

# Part 6: Hyperparameter experimentation (1 point)
Finally choose your best model from the 3 available. Choose 2 tunnable hyperparameters and 3 values for each one. Perform a 3X3 grid search, report your results and discuss why do you think you obtain your best results with the final model.

# Report
Please upload to your repository a PDF file named `lastname_hw3.pdf`. Use a text editor such as latex but use **single column formating**.
