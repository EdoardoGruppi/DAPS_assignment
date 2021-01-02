# Import packages
from pandas import DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn


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
    figure = plt.figure(figsize=(10, 4))
    for index, col in enumerate(columns):
        figure.add_subplot(1, len(columns), index+1)
        sn.boxplot(y=col, data=df)
    figure.tight_layout()
    plt.show()
    # Plot the pairwise joint distributions
    for label in hue:
        sn.pairplot(df[columns].join(df[label]), hue=label)
        plt.show()


def plot_rolling(series, window=12):
    df = DataFrame(series.values, columns=['data'])
    df['z_data'] = (df['data'] - df.data.rolling(window=window).mean()) / df.data.rolling(window=window).std()
    df['zp_data'] = df['z_data'] - df['z_data'].shift(12)
    sn.set()
    fig, ax = plt.subplots(3, sharex='all', figsize=(12, 9))
    ax[0].plot(df.index, df.data, label='raw data')
    ax[0].plot(df.data.rolling(window=window).mean(), label="rolling mean")
    ax[0].plot(df.data.rolling(window=window).std(), label="rolling std (x10)")
    ax[0].legend()
    ax[1].plot(df.index, df.z_data, label="de-trended data")
    ax[1].plot(df.z_data.rolling(window=window).mean(), label="rolling mean")
    ax[1].plot(df.z_data.rolling(window=window).std(), label="rolling std (x10)")
    ax[1].legend()
    ax[2].plot(df.index, df.zp_data, label=f"{window} lag differenced de-trended data")
    ax[2].plot(df.zp_data.rolling(window=window).mean(), label="rolling mean")
    ax[2].plot(df.zp_data.rolling(window=window).std(), label="rolling std (x10)")
    ax[2].legend()
    plt.tight_layout()
    plt.show()


