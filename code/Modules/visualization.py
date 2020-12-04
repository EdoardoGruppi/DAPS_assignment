# Import packages
import matplotlib.pyplot as plt
import seaborn as sn


def plot_column(data, target_column):
    sn.set()
    data[target_column].plot()
    plt.title('Daily {} for the Microsoft stock'.format(target_column))
    plt.show()
