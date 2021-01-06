# Import packages
from pandas import DataFrame, set_option
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn

set_option('display.max_columns', None)


def change_format(dataset):
    dataframe = dataset.copy()
    columns = dataframe.columns
    dataframe['Day'] = dataframe.index.day
    dataframe['Month'] = dataframe.index.month
    dataframe['Year'] = dataframe.index.year
    dataframe['Quarter'] = dataframe.index.quarter
    dataframe['WeekDay'] = dataframe.index.weekday
    dataframe = dataframe.reset_index(drop=True)
    return dataset, dataframe, columns


def attributes_visualization(df, columns, hue=None):
    print('Preview of data:\n', df.tail(3), '\nInfo:\n', df.info(), '\nDistribution:\n', df.describe().T)
    sn.set()
    figure = plt.figure(figsize=(21, 12))
    for index, col in enumerate(columns):
        figure.add_subplot(1, len(columns), index + 1)
        sn.boxplot(y=col, data=df, linewidth=3.5)
        figure.tight_layout()
    plt.show()
    # Plot the pairwise joint distributions
    if hue is not None:
        for label in hue:
            sn.pairplot(df[columns].join(df[label]), hue=label)
    else:
        sn.pairplot(df[columns])
    figure.tight_layout()
    plt.show()


def plot_rolling(series, short_window=20, long_window=50):
    fig, axes = plt.subplots(2, 1, sharex='all', figsize=(25, 15))
    sn.lineplot(data=series.rolling(window=short_window, min_periods=1).mean(), ax=axes[0])
    sn.lineplot(data=series.ewm(span=short_window, adjust=False).mean(), ax=axes[0])
    sn.lineplot(data=series, ax=axes[0])
    sn.lineplot(data=series.rolling(window=long_window, min_periods=1).mean(), ax=axes[1])
    sn.lineplot(data=series.ewm(span=long_window, adjust=False).mean(), ax=axes[1])
    sn.lineplot(data=series, ax=axes[1])
    plt.tight_layout()
    plt.show()
