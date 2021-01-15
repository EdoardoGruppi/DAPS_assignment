# Import packages
from pandas import DataFrame, set_option
from numpy import percentile as compute_percentile
import matplotlib.pyplot as plt
import seaborn as sn
from numpy import sqrt, abs, round
from scipy.stats import norm
from Modules.utilities import plot_attribute_properties
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


def custom_test_1(series, second_series, significance_level=0.05, threshold=0.15, test=0):
    """
    H0: The sentiment extracted from Twitter posts does not affect the daily percentage variation of the stock closing
    price.

    :param test:
    :param threshold:
    :param series:
    :param second_series:
    :param significance_level:
    :return:
    """
    dataframe = DataFrame([series, second_series]).transpose()
    sample1 = dataframe[dataframe.iloc[:, 1] > threshold].iloc[:, 0]
    sample2 = dataframe[dataframe.iloc[:, 1] <= threshold].iloc[:, 0]
    # Compute mean, std and size of each sample
    mean1, mean2 = sample1.mean(), sample2.mean()
    sigma1, sigma2 = sample1.std(), sample2.std()
    size1, size2 = sample1.shape[0], sample2.shape[0]
    two_samples_t_test(mean1, mean2, sigma1, sigma2, size1, size2, significance_level, test)


def custom_test_2(series, second_series, significance_level=0.05, percentile=90, test=0):
    """
    H0: The higher 20% of the sentiment extracted from Twitter posts does not affect the daily percentage
    variation of the stock closing price.

    :param test:
    :param percentile:
    :param series:
    :param second_series:
    :param significance_level:
    :return:
    """
    dataframe = DataFrame([series, second_series]).transpose()
    percentile = compute_percentile(second_series, percentile)
    sample1 = dataframe[dataframe.iloc[:, 1] > percentile].iloc[:, 0]
    sample2 = dataframe[dataframe.iloc[:, 1] <= percentile].iloc[:, 0]
    # Compute mean, std and size of each sample
    mean1, mean2 = sample1.mean(), sample2.mean()
    sigma1, sigma2 = sample1.std(), sample2.std()
    size1, size2 = sample1.shape[0], sample2.shape[0]
    two_samples_t_test(mean1, mean2, sigma1, sigma2, size1, size2, significance_level, test)


def two_samples_t_test(mean1, mean2, sigma1, sigma2, size1, size2, significance_level=0.05, test=0):
    overall_sigma = sqrt(sigma1 ** 2 / size1 + sigma2 ** 2 / size2)
    z_statistic = round((mean1 - mean2) / overall_sigma, 5)
    if test == 0:
        # Two tails -> H0:x1=x2 H1:x1!=x2
        print('\nTwo Tails test. H0 is x1=x2 and H1 is x1!=x2')
        p_value = round(2 * (1 - norm.cdf(abs(z_statistic))), 5)
    elif test == 1:
        # One tail right -> H0:x1>x2 H1:x1<=x2
        print('\nOne Tail test. H0 is x1>x2 and H1 is x1<=x2')
        p_value = round((1 - norm.cdf(z_statistic)), 5)
    else:
        # One tail left -> H0:x1<x2 H1:x1>=x2
        print('\nOne Tail test. H0 is x1<x2 and H1 is x1>=x2')
        p_value = round((norm.cdf(z_statistic)), 5)
    if p_value < significance_level:
        print(f'Statistic:{z_statistic} - P-value:{p_value} - Reject Null Hypothesis')
    else:
        print(f'Statistic:{z_statistic} - P-value:{p_value} - Do Not Reject Null Hypothesis')


def percentage_change(dataframe, cols):
    columns = dataframe.columns
    columns = [item for item in columns if item not in cols]
    dataframe_pct = dataframe.copy()
    for col in columns:
        dataframe_pct[col] = dataframe[col].pct_change(periods=1).fillna(0)
    return dataframe_pct
