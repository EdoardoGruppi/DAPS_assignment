# Import packages
import matplotlib.pyplot as plt
import seaborn as sn


def plot_dataframe(data, target_column=None):
    sn.set()
    if target_column is not None:
        data[target_column].plot()
        plt.title('Daily {} report for Microsoft stocks'.format(target_column.split('. ')[-1]))
    else:
        data.plot()
        plt.title('Daily report for Microsoft stocks')
    plt.show()
