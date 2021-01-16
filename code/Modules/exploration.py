# Import packages
from pandas import DataFrame, set_option
from numpy import percentile as compute_percentile
import matplotlib.pyplot as plt
import seaborn as sn
from numpy import sqrt, abs, round
from scipy.stats import norm
set_option('display.max_columns', None)


def change_format(dataset):
    """
    Obtains a tidy format dataframe from a copy of the dataset given.

    :param dataset: dataset to transform under a tidy format.
    :return: the original dataset, a copy under a tidy format and the list of columns of the original dataset.
    """
    # Copy the dataframe to avoid any kind of conflict
    dataframe = dataset.copy()
    # Get the list of the columns names of the dataset
    columns = dataframe.columns
    dataframe['Day'] = dataframe.index.day
    dataframe['Month'] = dataframe.index.month
    dataframe['Year'] = dataframe.index.year
    dataframe['Quarter'] = dataframe.index.quarter
    dataframe['WeekDay'] = dataframe.index.weekday
    dataframe = dataframe.reset_index(drop=True)
    return dataset, dataframe, columns


def attributes_visualization(df, columns, hue=None):
    """
    Plots the univariate box-plots of each attribute of the dataset passed along with a scatter plot matrix where the
    data points are coloured according to the day, the weekday, the month or the year in which they are measured.

    :param df: dataframe to visualize. If hue is not None it must be in a tidy format.
    :param columns: columns to consider in both the plots.
    :param hue: define how to separate the data points in the scatter plots. It can be 'Day', 'Month', 'Year', 'Quarter'
        and/or 'WeekDay'. default_value=None
    :return:
    """
    # Print some information about the dataset
    print(f'\nPreview of data:\n{df.tail(3)} \n\nDistribution:\n{df.describe().T}')
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
    H0: A specific range of values from the first series does not affect the daily percentage variation of the second
    series. H1: the opposite.

    :param test: define the test to execute. 0 means two tails two samples T-test; 1 means one tail two samples T-test;
        any other value means a one tail two samples T-test on the other side. default_value=0
    :param threshold: threshold to divide the population in two samples. default_value=0.15
    :param series: first series.
    :param second_series: second series.
    :param significance_level: superior limit of the p-value to reject H0. default_value=0.05
    :return:
    """
    # Obtain a dataframe with the series passed
    dataframe = DataFrame([series, second_series]).transpose()
    # Sample 1 contains only the first series values related to the second series values that are greater than the
    # threshold. The opposite for the second sample.
    sample1 = dataframe[dataframe.iloc[:, 1] > threshold].iloc[:, 0]
    sample2 = dataframe[dataframe.iloc[:, 1] <= threshold].iloc[:, 0]
    # Compute mean, std and size of each sample
    mean1, mean2 = sample1.mean(), sample2.mean()
    sigma1, sigma2 = sample1.std(), sample2.std()
    size1, size2 = sample1.shape[0], sample2.shape[0]
    # Perform the T-test required
    two_samples_t_test(mean1, mean2, sigma1, sigma2, size1, size2, significance_level, test)


def custom_test_2(series, second_series, significance_level=0.05, percentile=90, test=0):
    """
    H0: A specific range of values from the first series does not affect the daily percentage variation of the second
    series. H1: the opposite.

    :param test: define the test to execute. 0 means two tails two samples T-test; 1 means one tail two samples T-test;
        any other value means a one tail two samples T-test on the other side. default_value=0
    :param percentile: percentile that divides the population in two samples. default_value=0.15
    :param series: first series.
    :param second_series: second series.
    :param significance_level: superior limit of the p-value to reject H0. default_value=0.05
    :return:
    """
    # Obtain a dataframe with the series passed
    dataframe = DataFrame([series, second_series]).transpose()
    percentile = compute_percentile(second_series, percentile)
    # Sample 1 contains only the first series values related to the second series values that do not belong to the
    # percentile given. The opposite for the second sample.
    sample1 = dataframe[dataframe.iloc[:, 1] > percentile].iloc[:, 0]
    sample2 = dataframe[dataframe.iloc[:, 1] <= percentile].iloc[:, 0]
    # Compute mean, std and size of each sample
    mean1, mean2 = sample1.mean(), sample2.mean()
    sigma1, sigma2 = sample1.std(), sample2.std()
    size1, size2 = sample1.shape[0], sample2.shape[0]
    # Perform the T-test required
    two_samples_t_test(mean1, mean2, sigma1, sigma2, size1, size2, significance_level, test)


def two_samples_t_test(mean1, mean2, sigma1, sigma2, size1, size2, significance_level=0.05, test=0):
    """
    Executes a two sample T-test with the statistic properties passed.

    :param mean1: mean of the first sample.
    :param mean2: mean of the second sample.
    :param sigma1: standard deviation of the first sample.
    :param sigma2: standard deviation of the second sample.
    :param size1: size of the first sample.
    :param size2: size of the second sample.
    :param significance_level: superior limit of the p-value to reject H0. default_value=0.05
    :param test: define the test to execute. 0 means two tails two samples T-test; 1 means one tail two samples T-test;
        any other value means a one tail two samples T-test on the other side. default_value=0
    :return:
    """
    # Compute Z-statistic
    overall_sigma = sqrt(sigma1 ** 2 / size1 + sigma2 ** 2 / size2)
    z_statistic = round((mean1 - mean2) / overall_sigma, 5)
    # Compute p-value from Z-statistic
    if test == 0:
        # Two tails -> H0:x1=x2 H1:x1!=x2
        print('Two Tails test. H0 is x1=x2 and H1 is x1!=x2')
        p_value = round(2 * (1 - norm.cdf(abs(z_statistic))), 5)
    elif test == 1:
        # One tail right -> H0:x1>x2 H1:x1<=x2
        print('One Tail test. H0 is x1>x2 and H1 is x1<=x2')
        p_value = round((1 - norm.cdf(z_statistic)), 5)
    else:
        # One tail left -> H0:x1<x2 H1:x1>=x2
        print('One Tail test. H0 is x1<x2 and H1 is x1>=x2')
        p_value = round((norm.cdf(z_statistic)), 5)
    # Reject or not the Null hypothesis
    if p_value < significance_level:
        print(f'Statistic:{z_statistic} - P-value:{p_value} - Reject Null Hypothesis')
    else:
        print(f'Statistic:{z_statistic} - P-value:{p_value} - Do Not Reject Null Hypothesis')


def percentage_change(dataframe, cols):
    """
    Computes the daily percentage variation of the attributes of the dataset passed.

    :param dataframe: original dataset to elaborate.
    :param cols: list of columns not to consider in the computation.
    :return: a copy of the original dataset with the daily percentage changes of the variables.
    """
    # Retrieve the name of all the variables
    columns = dataframe.columns
    # Consider only the variables not in cols
    columns = [item for item in columns if item not in cols]
    dataframe_pct = dataframe.copy()
    for col in columns:
        # Compute the daily percentage variation
        dataframe_pct[col] = dataframe[col].pct_change(periods=1).fillna(0)
    return dataframe_pct
