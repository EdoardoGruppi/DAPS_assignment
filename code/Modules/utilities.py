# Import packages
import matplotlib.pyplot as plt
import seaborn as sn
import os
from pandas import read_csv, read_pickle
from Modules.config import *
import mplfinance as mpf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import numpy as np


def detect_univariate_outlier(dataframe):
    outlier_z_score = []
    outlier_iqr_score = []
    for column in dataframe:
        data = dataframe[column]
        sn.set()
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 7, 7]})
        fig.suptitle(f"BoxPlot, ScatterPlot and z-score of {company} {column}")
        sn.boxplot(data=data, ax=axes[0])
        # Iqr outliers
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        minimum = q1 - 1.5 * (q3 - q1)
        maximum = q3 + 1.5 * (q3 - q1)
        outlier_iqr_score.append(dataframe.index[np.where((data < minimum) | (data > maximum))])
        sn.scatterplot(x=dataframe.index, y=data, hue=(data < minimum) | (data > maximum), ax=axes[1], legend=False)
        # Z score outliers
        z = np.abs(stats.zscore(data))
        outlier_z_score.append(dataframe.index[np.where(z > 3)])
        sn.scatterplot(x=dataframe.index, y=z, hue=z > 3, ax=axes[2], legend=False)
        # Z score outliers
        plt.show()
    return outlier_z_score, outlier_iqr_score


def ohlc_chart(data, start=starting_date, end=ending_date, candle_size='W', volume=False):
    # Change the names of the columns.
    data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '5. adjusted close': 'Close'})
    data = data[start:end].resample(candle_size).mean()
    # Plot candlestick.
    new_style = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'axes.grid': True})
    mpf.plot(data, type='candle', ylabel='Price ($)', volume=volume, mav=(5, 10), figsize=(30, 7), style=new_style,
             scale_padding=0.05, xrotation=0, savefig=dict(fname="ohlc.png", bbox_inches="tight"))
    mpf.show()


def decompose_series(series, period=30):
    sn.set()
    result = seasonal_decompose(series.values, model='multiplicative', period=period)
    result_2 = seasonal_decompose(series.values, model='additive', period=period)
    result.plot()
    print(f'Multiplicative: [mean: {result.seasonal.mean()}, max:{result.seasonal.max()}, min:{result.seasonal.min()}]')
    plt.show()
    result_2.plot()
    print(f'Additive: [mean: {result_2.seasonal.mean()}, max:{result_2.seasonal.max()}, min:{result_2.seasonal.min()}]')
    plt.show()


def plot_dataframe(data, target_column=None):
    sn.set()
    if target_column is not None:
        data[target_column].plot()
        plt.title(f'Daily {target_column} report for {company} stocks')
    else:
        data.plot()
        plt.title(f'Daily report for {company} stocks')
    plt.show()


def count_missing_values(dataframe):
    missing_values = dataframe.isnull().sum()
    print(missing_values)
    return missing_values.sum() == 0


def csv2pickle(filename, remove=True):
    path = os.path.join(base_dir, filename)
    df = read_csv(path, sep=',')
    if remove:
        os.remove(path)
    filename = filename.split('.')[0] + '.pkl'
    dataframe_path = os.path.join(base_dir, filename)
    df.to_pickle(dataframe_path)
    return dataframe_path


def pickle2csv(filename, remove=False):
    path = os.path.join(base_dir, filename)
    df = read_pickle(path)
    if remove:
        os.remove(path)
    filename = filename.split('.')[0] + '.csv'
    dataframe_path = os.path.join(base_dir, filename)
    df.to_csv(dataframe_path)
    return dataframe_path
