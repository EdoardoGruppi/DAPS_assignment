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

- **Os** provides a portable way of using operating system dependent functionality.

- **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- **Sklearn** offers simple and efficient tools for predictive data analysis.

- **Seaborn** is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics.

- **Alpha_vantage** delivers a free API for real time financial data and most used finance indicators in a simple json or pandas format. This module implements a python interface to the free API provided by Alpha Vantage.

- **Pmdarima** pmdarima brings R’s auto.arima function to Python. Pmdarima is written in Python and Cython and provides an easy-to-use set of functions and classes.

- **Fbprophet** is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

- **Mplfinance** provides several utilities for the visualization, and visual analysis, of financial data.

- **Statsmodels** statsmodels is a Python package that provides a complement to scipy for statistical computations including descriptive statistics and estimation and inference for statistical models.

- **Scipy** is an open-source software for mathematics, science, and engineering. The SciPy library strictly depends on NumPy.

- **Pylab** is a procedural interface to the Matplotlib object-oriented plotting library. Actually, PyLab is not a package. It is a module that gets installed alongside Matplotlib.

- **Re** provides regular expression matching operations similar to those found in Perl. Built-in package.

- **Datetime** comes built into Python, so there is no need to install it externally. It supplies classes to work with date and time.

- **Flair** is a very simple framework for state-of-the-art NLP. The package makes available different useful pre-trained models.

- **Nltk** is a Python package for natural language processing.

- **Covid19dh** provides a unified dataset by collecting worldwide fine-grained case data, merged with exogenous variables helpful for a better understanding of COVID-19.

- **Pyod** is a comprehensive and scalable Python toolkit for detecting outlying objects in multivariate data.

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are realised. More precisely, it is responsible to call functions from other files in order to divide the datasets provided, pre-process images and instantiate, train and test models.

**config.py** makes available all the global variables used in the project.

**covid_data.py** contains functions to collect and pre-process pandemic data.

**data_gatherer.py** is a script useful to retrieve and save the datasets employed in the project.

**news_data.py** provides a function to gather messages on the Twitter platform from a list of selected sources.

**sentiment_analysis.py** offers functions to perform natural language processing using either a flair pre-trained model or vader.

**stock_data.py** contains functions to collect and pre-process stock data through the alpha_vantage API.

**twitter_data.py** delivers functionalities to create queries, collect data from Twitter using Twint and pre-process tweets.

**utilities.py** provides several functions useful to analyse data through visualization and statistic. Moreover, additional functionalities are made available to simply manage and handle data.

**arima.py** contains all the necessary to predict future values of a time series through a model of the ARIMA's family.

**prophet.py** offfers the possibility to forecast future values of a time series via the Facebook Prophet model.

**\_Additional_code folder** includes some .py files useful for the code devolepment as well as to report the most noteworthy experiments conducted during the project.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is a cross platform integrated development environment (IDE) for Python programmers. The choice
fell on it because of its ease of use while remaining one of the most advanced working environments.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular machine learning libraries and it offers GPUs where you can execute the code as well.
