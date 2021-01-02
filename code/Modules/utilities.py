# Import packages
import matplotlib.pyplot as plt
import seaborn as sn
import os
from pandas import read_csv, read_pickle, DataFrame, concat, Series
from Modules.config import *
import mplfinance as mpf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.api import qqplot
from scipy import stats
import numpy as np
import pylab
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from pyod.models import feature_bagging, hbos, iforest, knn, mcd
from pmdarima.arima import ADFTest, ndiffs


def detect_univariate_outlier(dataframe, cap=None, nan=False):
    """
    Detects univariate outliers for each attribute belonging to the dataframe passed. The function displays three
    distinct plots (boxplot, scatter plot with iqr outliers, scatter plot after calculating z-score) where outliers are
    displayed with red dots. Once the outliers are detected the function can cap them (according to the maximum value
    that does not correspond to outliers with either iqr score or Z-score) or replace their values with NaN.

    :param dataframe: dataset to analyse.
    :param cap: if 'iqr' or 'z_score' operates on the outliers detected with iqr score and z-score respectively.
        default_value=None
    :param nan: if it is True the outlier selected according to the cap parameter are replaced by NaN, otherwise they
        are capped with the maximum "legit" value. default_value=False
    :return:
    """
    # Outliers rows are not deleted. Alternatives considered are: doing nothing, capping or replacing their values.
    for column in dataframe:
        data = dataframe[column]
        # Create the figure
        sn.set()
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 7, 7]})
        fig.suptitle(f"BoxPlot, ScatterPlot and Z-score of {company} {column}")
        sn.boxplot(data=data, ax=axes[0])
        # Compute iqr outliers and their position
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        minimum = q1 - 1.5 * (q3 - q1)
        maximum = q3 + 1.5 * (q3 - q1)
        outlier_iqr_loc = dataframe.index[np.where((data < minimum) | (data > maximum))]
        g = sn.scatterplot(x=dataframe.index, y=data, hue=(data < minimum) | (data > maximum), ax=axes[1], legend=False)
        g.set(xlabel=None, ylabel=None)
        # Compute Z-score outliers and their position
        z = np.abs(stats.zscore(data))
        outlier_z_loc = dataframe.index[np.where(z > 3)]
        g = sn.scatterplot(x=dataframe.index, y=z, hue=z > 3, ax=axes[2], legend=False)
        g.set(xlabel=None, ylabel=None)
        plt.show()
        # Change the outlier values if required and if there are outliers
        if cap == 'z_score' and len(outlier_z_loc) > 0:
            if nan:
                # Replace with Nan values
                data[outlier_z_loc] = float('nan')
            else:
                # Find all the elements that do not correspond to outliers
                dropped_outlier_dataset = data[np.setdiff1d(data.index, outlier_z_loc)]
                data[outlier_z_loc] = np.max(dropped_outlier_dataset)
            # Save the changed values in the dataframe passed
            dataframe[column] = data
        if cap == 'iqr' and len(outlier_iqr_loc) > 0:
            if nan:
                # Replace with Nan values
                data[outlier_iqr_loc] = float('nan')
            else:
                # Find all the elements that do not correspond to outliers
                dropped_outlier_dataset = data[np.setdiff1d(data.index, outlier_iqr_loc)]
                data[outlier_iqr_loc] = np.max(dropped_outlier_dataset)
            # Save the changed values in the dataframe passed
            dataframe[column] = data


def multivariate_visualization(dataframe):
    """
    Plots the scatter plot and the correlation matrices of the dataset attributes.

    :param dataframe: dataset to analyse.
    :return:
    """
    sn.set(font_scale=1)
    # Use number instead of columns to get a tidier layout
    labels = np.arange(dataframe.shape[1])
    fig, axes = plt.subplots(1, 2, sharex='all', figsize=(8, 4))
    # Pearson correlation matrix
    pearson_matrix = dataframe.corr(method='pearson')
    sn.heatmap(pearson_matrix, annot=True, linewidths=2, cmap='GnBu_r', cbar=False, square=True, ax=axes[0],
               xticklabels=labels, yticklabels=labels, vmax=1, vmin=-1)
    # Spearman correlation matrix
    corr_matrix = dataframe.corr(method='spearman')
    sn.heatmap(corr_matrix, annot=True, linewidths=2, cmap='GnBu_r', cbar=False, square=True, ax=axes[1],
               xticklabels=labels, yticklabels=labels, vmax=1, vmin=-1)
    plt.show()
    # Scatter plot matrix
    g = sn.pairplot(dataframe, plot_kws=dict(s=10), diag_kind='hist', diag_kws=dict(kde=True, bins=50))
    g.map_upper(sn.kdeplot, levels=4)
    plt.show()


def dataset_division(dataframe, valid_size=30, percentage=False):
    """
    Divides the dataset received in three parts: train, validation and test. The validation size can be expressed in
    number of observations or in percentage.

    :param dataframe: dataset to segment.
    :param valid_size: dimension of validation set in number of observations or percentage. In the latter case set the
        percentage parameter to True. default_value=False
    :param percentage: if True valid_size is expressed as a percentage. default_value=False.
    :return: the three parts obtained after the splitting
    """
    # Train part is defined in the global variables saved in config.py
    train = dataframe.loc[starting_date:ending_date]
    # Compute how many samples are reserved to training
    if percentage:
        train_samples = round(train.shape[0] * (1 - valid_size))
    else:
        train_samples = train.shape[0] - valid_size
    # Divide the dataset in consecutive parts given that it is a time-series
    valid = train.iloc[train_samples:, :]
    train = train.iloc[:train_samples, :]
    test = dataframe.loc[starting_test_period:ending_test_period]
    return train, valid, test


def combine_dataset(datasets):
    """
    Combines dataframes together. The main dataframe must be passed as first. The other datasets are concatenated
    horizontally.

    :param datasets: list of dataframes to concatenate.
    :return: the unified dataset.
    """
    # The first dataframe is the most important since it defines the number and which observations to keep.
    dataframe = datasets[0]
    del datasets[0]
    # Concatenate one dataframe at a time
    for dataset in datasets:
        dataframe = dataframe.join(dataset, how='left')
    # Drop all the observations that exceed the entire period observed
    dataframe = dataframe[starting_date:ending_test_period]
    # Replace the missing values created by this step with 0
    dataframe = dataframe.fillna(0)
    return dataframe


def shift_dataset(dataset, column='Close'):
    """
    Shift the columns of the dataset so that the values of the previous day of all the variables can be used to predict
    the next day of the variable of interest. The shifting is needed since many forecasting techniques when working with
    exogenous variables require the future values of those. To understand better it is possible to explore the formulas
    related to these techniques.

    :param dataset: dataset to modify.
    :param column: column or list of columns not to include during the shifting. default_value='Close'
    :return: the new dataset.
    """
    # Time-series shifted is needed due to how fbprophet and arima make predictions.
    columns = [col for col in dataset.columns if col != column]
    dataset[columns] = dataset[columns].shift(1)
    # Fill the new Nan in the first line
    dataset = dataset.bfill()
    return dataset


def transform_dataset(train, valid, test, algorithm='pca', n_components=2, kernel='rbf', perplexity=5, reduction=True):
    """
    Transforms the dataset computing several operations. Firstly, it excludes by any computation the time series to
    predict. Secondly, if dimension reduction is required before applying the algorithm selected (pca, kernel pca,
    t-sne) it normalizes the input in the range [0,1]. Normalization before dimension reduction is crucial. Then, in
    both the cases (reduction requested or not requested) it brings all the features to the same scale, i.e. the scale
    of the series to predict.

    :param train: train dataset.
    :param valid: validation dataset.
    :param test: test dataset.
    :param algorithm: dimensionality reduction algorithm to apply ('pca','pca_kernel','t-sne'). It is valid only if
        reduction=True. default_value='pca'
    :param n_components: number of components to keep after computing one of the algorithms. It can also be used to
        express instead the variance to kept. default_value=2
    :param kernel: valid only if reduction is True and algorithm is 'kernel_pca'. default_value='rbf'
    :param perplexity: valid only if reduction is True and algorithm is 't-sne'. default_value=5
    :param reduction: if True the dimensionality reduction is applied. default_value=False
    :return: train, validation and test set normalized and if required also reduced.
    """
    # Dictionary of the dimensionality reduction algorithms
    function = {'pca': PCA(n_components=n_components),
                'kernel_pca': KernelPCA(n_components=n_components, kernel=kernel),
                't-sne': TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)}
    # Exclude the main column from the computations
    train_data = train.drop(['Close'], axis=1)
    valid_data = valid.drop(['Close'], axis=1)
    test_data = test.drop(['Close'], axis=1)
    # Apply dimensionality reduction exclusively when it is True
    if reduction:
        # Select the algorithm
        model = function[algorithm]
        # Normalization before applying dimensionality reduction
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)
        test_data = scaler.transform(test_data)
        # Apply dimensionality reduction
        train_data = model.fit_transform(train_data)
        valid_data = model.transform(valid_data)
        test_data = model.transform(test_data)
    # Scale all the other features to the scale of the main column
    scaler = MinMaxScaler(feature_range=(train.Close.min(), train.Close.max()))
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    # Recreate the dataframes from the array returned by the various algorithms involved
    train = concat([DataFrame(train_data, index=train.index), train['Close']], axis=1)
    valid = concat([DataFrame(valid_data, index=valid.index), valid['Close']], axis=1)
    test = concat([DataFrame(test_data, index=test.index), test['Close']], axis=1)
    # Transform in string all the names f the columns to avoid conflicts later.
    train.columns = [str(col) for col in train.columns]
    valid.columns = [str(col) for col in valid.columns]
    test.columns = [str(col) for col in test.columns]
    return train, valid, test


def metrics(y, y_hat):
    """
    Computes metrics (MAPE, RMSE, CORR, R2, MAE, MPE, MSE) to evaluate models performance.

    :param y: the true values of the test dataset.
    :param y_hat: the predicted values.
    :return:
    """
    # Compute errors
    d = y - y_hat
    # Compute and print metrics
    mse_f = np.mean(d ** 2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    mape = np.mean(np.abs(y_hat - y) / np.abs(y))
    mpe = np.mean((y_hat - y) / y)
    corr = np.corrcoef(y_hat, y)[0, 1]
    r2_f = 1 - (sum(d ** 2) / sum((y - np.mean(y)) ** 2))
    print('\nResults by manual calculation:\n',
          f'- MAPE: {mape:.4f} \n - RMSE: {rmse_f:.4f} \n - CORR: {corr:.4f} \n - R2: {r2_f:.4f}\n',
          f'- MAE: {mae_f:.4f} \n - MPE: {mpe:.4f} \n - MSE: {mse_f:.4f}')


def residuals_properties(residuals):
    """
    Computes statistical values and displays plots to evaluate how the models fitted the training dataset. The residuals
    in a time series model are what is left over after fitting a model.

    :param residuals: residuals of the model.
    :return:
    """
    # Compute mean, median, skewness, kurtosis and durbin statistic
    residuals = residuals[1:]
    mean = residuals.mean()
    median = np.median(residuals)
    # skewness = 0 : normally distributed.
    # skewness > 0 : more weight in the left tail of the distribution. Long right tail. Median before mean.
    # skewness < 0 : more weight in the right tail of the distribution. Long left tail. Median after mean.
    skew = stats.skew(residuals)
    # Kurtosis is the degree of the peak of a distribution.
    # 3 it is normal, >3 higher peak, <3 lower peak
    kurtosis = stats.kurtosis(residuals)
    # Durbin-Watson statistic equal to  2.0 means no auto-correlation.
    # Values between 0 and 2 indicate positive and values between 2 and 4 indicate negative auto-correlation.
    durbin = durbin_watson(residuals)
    # Shapiro-Wilk quantifies how likely it is that the data was drawn from a Gaussian distribution.
    # Null hypothesis: the sample is normally distributed
    shapiro = stats.shapiro(residuals)[1]
    # Anderson-Darling test null hypothesis: the sample follows the normal distribution
    anderson = stats.normaltest(residuals)[1]
    print(f'\nResidual information:\n - Mean: {mean:.4f} \n - Median: {median:.4f} \n - Skewness: {skew:.4f} '
          f'\n - Kurtosis: {kurtosis:.4f}\n - Durbin: {durbin:.4f}',
          f'\n - Shapiro p-value: {shapiro:.4f}\n - Anderson p-value: {anderson:.4f}\n')
    # Create plots
    sn.set()
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    # Compute standardized residuals
    residuals = (residuals - np.nanmean(residuals)) / np.nanstd(residuals)
    # First picture: q-q plot
    # Keep only not NaN residuals.
    residuals_non_missing = residuals[~(np.isnan(residuals))]
    qqplot(residuals_non_missing, line='s', ax=axes[0])
    axes[0].set_title('Normal Q-Q')
    # Second picture: simple plot of standardized residuals
    x = np.arange(0, len(residuals), 1)
    sn.lineplot(x=x, y=residuals, ax=axes[1])
    axes[1].set_title('Standardized residual')
    # Third picture: comparison between residual and gaussian distribution
    kde = stats.gaussian_kde(residuals_non_missing)
    x_lim = (-1.96 * 2, 1.96 * 2)
    x = np.linspace(x_lim[0], x_lim[1])
    axes[2].plot(x, stats.norm.pdf(x), label='Normal (0,1)', lw=2)
    axes[2].plot(x, kde(x), label='Residuals', lw=2)
    axes[2].set_xlim(x_lim)
    axes[2].legend()
    axes[2].set_title('Histogram plus estimated density')
    # Last pictures: residuals auto-correlation plots
    plot_acf(residuals, ax=axes[3], lags=30)
    plot_pacf(residuals, ax=axes[4], lags=30)
    plt.show()


def check_stationarity(time_series):
    """
    Performs the Augmented Dickey-Fuller test wherein the null hypothesis is: data is not stationary. Adopting an alpha
    value of 0.05, the null hypothesis will be rejected only when the confidence is greater than 95%. This function also
    differences until it can be considered stationary with 95% or more of confidence.

    :param time_series: series to analyse with the ADF test.
    :return:
    """
    # Make sure that the original time series is not modified
    series = time_series.copy()
    print('Results of Dickey-Fuller Test:')
    adf_test = ADFTest(alpha=0.05)
    diff_order = 0
    while True:
        # Compute the ADF test. It returns the p-value and if differencing is needed
        results, should = adf_test.should_diff(series)
        print(f'Differencing order: {diff_order} - P-value: {results:.4f} - Stop: {not should}')
        if should:
            # If it is not already stationary, apply differencing of one order above
            diff_order += 1
            series = series.diff(periods=diff_order).bfill()
        else:
            break


def plot_auto_correlation(series, title=None):
    """
    Plots the auto correlation functions of the series provided.

    :param series: time series to analyse.
    :param title: title (optional) of the picture. default_value=None
    :return:
    """
    sn.set()
    fig, axes = plt.subplots(1, 2, sharey='all', figsize=(13, 5))
    if title is not None:
        fig.suptitle(title)
    plot_acf(series, ax=axes[0])
    plot_pacf(series, ax=axes[1])
    plt.show()


def granger_test(dataframe, target_column, max_lag=1, test='ssr_ftest'):
    """
    Performs granger test on the time series passed. Null hypothesis: the second time series, x2, does NOT Granger
    cause the time series of interest, x1. Grange causality means that past values of x2 have a statistically
    significant effect on the current value of x1.

    :param dataframe: dataset to analyse.
    :param target_column: time series x1.
    :param max_lag: maximum number of lags to consider. default_value=5
    :param test: which test to keep between the four computed by the stat function. default_value='ssr_ftest'
    :return:
    """
    print('\n\nGranger Test:')
    results = []
    columns = [col for col in dataframe.columns if col != target_column]
    for col_name in columns:
        dictionary = grangercausalitytests(dataframe[[target_column, col_name]], maxlag=max_lag)
        results.append(Series([item[0][test][1] for item in dictionary.values()], name=col_name))
    results = concat(results, axis=1)
    print(results)


def decompose_series(series, period=None, mode='multiplicative'):
    """
    Decomposes a series using moving averages.

    :param series: time series to decompose.
    :param period: Period of the series. Must be used if x is not a pandas object. default_value=None
    :param mode: Type of seasonal component ('additive', 'multiplicative'). default_value='multiplicative'
    :return:
    """
    sn.set()
    result = seasonal_decompose(series, model=mode, period=period)
    result.plot().set_figwidth(15)
    print(f'{mode}: [mean: {result.seasonal.mean()}, max:{result.seasonal.max()}, min:{result.seasonal.min()}]')
    plt.show()
    residuals_properties(result.resid.dropna())


def ohlc_chart(data, start=starting_date, end=ending_date, candle_size='W', volume=False):
    """
    Plots the open, high, low, close chart of the company.

    :param data: stock dataset.
    :param start: the first date to consider while creating the chart. default_value=starting_date
    :param end: the last date to consider while creating the chart. default_value=ending_date
    :param candle_size: dimension of the candles. default_value='W'
    :param volume: if True the volume is also reported below the ohlc chart. default_value=False
    :return:
    """
    data = data[start:end].resample(candle_size).mean()
    # Plot candlestick.
    new_style = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'axes.grid': True})
    mpf.plot(data, type='candle', ylabel='Price ($)', volume=volume, mav=(5, 10), figsize=(30, 7), style=new_style,
             scale_padding=0.05, xrotation=0, savefig=dict(fname="ohlc.png", bbox_inches="tight"))
    mpf.show()


def csv2pickle(filename, remove=True):
    """
    Transforms a csv file in a pickle file.

    :param filename: name of the csv file.
    :param remove: if True the original file is deleted. default_value=True
    :return: the path of the new pickle file.
    """
    # Create the path
    path = os.path.join(base_dir, filename)
    # Read the dataframe
    df = read_csv(path, sep=',')
    if remove:
        os.remove(path)
    # Create the new path using a new filename obtained from the previous one
    filename = filename.split('.')[0] + '.pkl'
    dataframe_path = os.path.join(base_dir, filename)
    # Save the dataframe in the new file format
    df.to_pickle(dataframe_path)
    return dataframe_path


def pickle2csv(filename, remove=False):
    """
    Transforms a pickle file in a csv file.

    :param filename: name of the pickle file.
    :param remove: if True the original file is deleted. default_value=False
    :return: the path of the new csv file.
    """
    # Create the path
    path = os.path.join(base_dir, filename)
    # Read the dataframe
    df = read_pickle(path)
    if remove:
        os.remove(path)
    # Create the new path using a new filename obtained from the previous one
    filename = filename.split('.')[0] + '.csv'
    dataframe_path = os.path.join(base_dir, filename)
    # Save the dataframe in the new file format
    df.to_csv(dataframe_path)
    return dataframe_path


def detect_seasonality(dataframe, column):
    """
    Displays two seasonal plots to detect recurrent patterns in data throughout years or months.

    :param dataframe: input data.
    :param column: target column to visualize.
    :return:
    """
    sn.set()
    fig, axes = plt.subplots(1, 2, figsize=(23, 5), gridspec_kw={'width_ratios': [2, 5]})
    sn.lineplot(data=dataframe, x='Month', y=column, hue='Year', legend='full', ax=axes[0])
    axes[0].set_title('Yearly seasonality plot')
    sn.lineplot(data=dataframe, x='Day', y=column, hue='Month', legend='full', ci=None, ax=axes[1])
    axes[1].set_title('Monthly seasonality plot')
    plt.legend()
    plt.show()


def plot_dataframe(data, target_column=None):
    """
    Plots one or all the columns of the dataframe passed.

    :param data: dataset to visualize.
    :param target_column: column to plot. If it is None all the columns of the dataframe are plotted. default_value=None
    :return:
    """
    sn.set()
    if target_column is not None:
        data[target_column].plot()
        plt.title(f'Daily {target_column} report for {company} stocks')
    else:
        data.plot()
        plt.title(f'Daily report for {company} stocks')
    plt.show()


def detect_multivariate_outlier(data, clf='iforest', contamination=0.03):
    """
    Detects the multivariate outliers inside a dataframe. It ranks all points by raw outlier scores and then mark the
    top %contamination as outliers.

    :param data: dataset to analyse.
    :param clf: algorithm or model adopted to detect outliers. default_value='iforest'
    :param contamination: percentage of contamination. default_value=0.03
    :return: the dates of the observations detected as multivariate outliers.
    """
    # Visualization difficult to realise when there are 4 or more features
    classifiers = {'hbos': hbos.HBOS, 'feature_bagging': feature_bagging.FeatureBagging, 'iforest': iforest.IForest,
                   'knn': knn.KNN, 'mcd': mcd.MCD}
    clf = classifiers[clf](contamination=contamination)
    # Fit detector
    clf.fit(data)
    # The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.
    labels = clf.labels_
    # The outlier scores of the training data. The higher, the more abnormal.
    scores = clf.decision_scores_
    index = np.where(labels == 1)
    return data.index[index]
