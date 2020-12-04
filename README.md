# Description of the project

[Project](https://github.com/EdoardoGruppi/DAPS_assignment) ~ [Guide](https://github.com/EdoardoGruppi/DAPS_assignment/blob/Secondary/code/Instructions.md)

## How to start

A comprehensive guide concerning how to run the code along with additional information is provided in the file [Instruction.md](https://github.com/EdoardoGruppi/DAPS_assignment/blob/Secondary/code/Instructions.md).

To first understand: which packages are required to the execution of the code, the role of each file or the software used read the Sections below.

## Packages required

The following list gathers all the packages needed to run the project code.
Please note that the descriptions provided in this subsection are taken directly from the package source pages. In order to have more details on them it is reccomended to directly reference to their official sites.

**Compulsory :**

- **Pandas** provides fast, flexible, and expressive data structures designed to make working with structured and time series data both easy and intuitive.

- **Numpy** is the fundamental package for array computing with Python.

- **Tensorflow** is an open source software library for high performance numerical computation. Its allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs). **Important**: Recently Keras has been completely wrapped within Tensorflow.

- **Pathlib** offers a set of classes to handle filesystem paths.

- **Shutil** provides a number of high-level operations on files and collections of files. In particular, functions are provided which support file copying and removal.

- **Os** provides a portable way of using operating system dependent functionality.

- **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- **Sklearn** offers simple and efficient tools for predictive data analysis.

- **Seaborn** is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process images and instantiate, train and test models.

ALTRO

**config.py** makes available all the global variables used in the project.

**pre_processing.py**

**results_visualization.py** exploits the seaborn and matplotlib libraries to plot the performance and learning curves of the training phase and to generate confusion matrices summarizing the models results.

**\_Additional_code folder**

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is a cross platform integrated development environment (IDE) for Python programmers. The choice
fell on it because of its ease of use while remaining one of the most advanced working environments.
