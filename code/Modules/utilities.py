# Import packages
import matplotlib.pyplot as plt
import seaborn as sn
import os
from pandas import read_csv
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


def csv2pickle(filename):
    path = os.path.join(base_dir, filename)
    df = read_csv(path, sep=',')
    os.remove(path)
    filename = filename.split('.')[0] + '.pkl'
    dataframe_path = os.path.join(base_dir, filename)
    df.to_pickle(dataframe_path)
    return dataframe_path
