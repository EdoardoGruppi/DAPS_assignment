# Import packages
import matplotlib.pyplot as plt
import seaborn as sn
import os
from pandas import read_csv, read_pickle
from Modules.config import *


def plot_dataframe(data, target_column=None):
    sn.set()
    if target_column is not None:
        data[target_column].plot()
        plt.title('Daily {} report for Microsoft stocks'.format(target_column.split('. ')[-1]))
    else:
        data.plot()
        plt.title('Daily report for Microsoft stocks')
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
