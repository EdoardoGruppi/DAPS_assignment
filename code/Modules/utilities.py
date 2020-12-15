# Import packages
import matplotlib.pyplot as plt
import seaborn as sn
import os
from pandas import read_csv, read_pickle, DataFrame, concat, Series
from Modules.config import *
import mplfinance as mpf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import numpy as np
import pylab
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from pyod.models import feature_bagging, hbos, iforest, knn, mcd


def detect_univariate_outlier(dataframe, cap='', nan=False):
    # Remove not considered. Possible alternatives are: doing nothing, capping or replacing.
    for column in dataframe:
        data = dataframe[column]
        sn.set()
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 7, 7]})
        fig.suptitle(f"BoxPlot, ScatterPlot and Z-score of {company} {column}")
        sn.boxplot(data=data, ax=axes[0])
        # Iqr outliers
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        minimum = q1 - 1.5 * (q3 - q1)
        maximum = q3 + 1.5 * (q3 - q1)
        outlier_iqr_loc = dataframe.index[np.where((data < minimum) | (data > maximum))]
        g = sn.scatterplot(x=dataframe.index, y=data, hue=(data < minimum) | (data > maximum), ax=axes[1], legend=False)
        g.set(xlabel=None, ylabel=None)
        # Z score outliers
        z = np.abs(stats.zscore(data))
        outlier_z_loc = dataframe.index[np.where(z > 3)]
        g = sn.scatterplot(x=dataframe.index, y=z, hue=z > 3, ax=axes[2], legend=False)
        g.set(xlabel=None, ylabel=None)
        plt.show()

        if cap == 'z_score' and len(outlier_z_loc) > 0:
            if nan:
                data[outlier_z_loc] = float('nan')
            else:
                dropped_outlier_dataset = data[np.setdiff1d(data.index, outlier_z_loc)]
                data[outlier_z_loc] = np.max(dropped_outlier_dataset)
            dataframe[column] = data
        if cap == 'iqr' and len(outlier_iqr_loc) > 0:
            if nan:
                data[outlier_iqr_loc] = float('nan')
            else:
                dropped_outlier_dataset = data[np.setdiff1d(data.index, outlier_iqr_loc)]
                data[outlier_iqr_loc] = np.max(dropped_outlier_dataset)
            dataframe[column] = data


def multivariate_visualization(dataframe):
    sn.set(font_scale=1)
    labels = np.arange(dataframe.shape[1])
    fig, axes = plt.subplots(1, 2, sharex='all', figsize=(10, 5))
    fig.suptitle(f"Correlation {company} dataset")
    pearson_matrix = dataframe.corr(method='pearson')
    sn.heatmap(pearson_matrix, annot=True, linewidths=2, cmap='GnBu_r', cbar=False, square=True, ax=axes[0],
               xticklabels=labels, yticklabels=labels, vmax=1, vmin=-1)
    corr_matrix = dataframe.corr(method='spearman')
    sn.heatmap(corr_matrix, annot=True, linewidths=2, cmap='GnBu_r', cbar=False, square=True, ax=axes[1],
               xticklabels=labels, yticklabels=labels, vmax=1, vmin=-1)
    plt.show()
    g = sn.pairplot(dataframe, plot_kws=dict(s=10), diag_kind='hist', diag_kws=dict(kde=True, bins=50))
    g.map_upper(sn.kdeplot, levels=4)
    plt.show()
    return


def detect_multivariate_outlier(data, clf='iforest', contamination=0.03):
    # It ranks all points by raw outlier scores and then mark the top %contamination as outliers.
    # Visualization difficult to realise when there are 4 or more features
    classifiers = {'hbos': hbos.HBOS, 'feature_bagging': feature_bagging.FeatureBagging, 'iforest': iforest.IForest,
                   'knn': knn.KNN, 'mcd': mcd.MCD}
    clf = classifiers[clf](contamination=contamination)
    # Fit detector.
    clf.fit(data)
    # The binary labels of the training data. 0 stands for inliers and 1 for outliers/anomalies.
    labels = clf.labels_
    # The outlier scores of the training data. The higher, the more abnormal.
    scores = clf.decision_scores_
    index = np.where(labels == 1)
    return data.index[index]


def dataset_division(dataframe, valid_size=0.2):
    train = dataframe.loc[starting_date:ending_date]
    train_samples = round(train.shape[0] * (1 - valid_size))
    valid = train.iloc[train_samples:, :]
    train = train.iloc[:train_samples, :]
    test = dataframe.loc[starting_test_period:ending_test_period]
    return train, valid, test


def transform_dataset(train, valid, test, algorithm='pca', n_components=2, kernel='rbf', perplexity=5):
    function = {'pca': PCA(n_components=n_components),
                'kernel_pca': KernelPCA(n_components=n_components, kernel=kernel),
                'tsne': TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)}
    train_data = train.drop(['Close'], axis=1)
    valid_data = valid.drop(['Close'], axis=1)
    test_data = test.drop(['Close'], axis=1)
    scaler = MinMaxScaler()
    columns = train_data.columns
    train_data = scaler.fit_transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    # Apply dimension reduction
    model = function[algorithm]
    train_data = model.fit_transform(train_data)
    valid_data = model.transform(valid_data)
    test_data = model.transform(test_data)
    train = concat([DataFrame(train_data, index=train.index), train['Close']], axis=1)
    valid = concat([DataFrame(valid_data, index=valid.index), valid['Close']], axis=1)
    test = concat([DataFrame(test_data, index=test.index), test['Close']], axis=1)
    train.columns = [str(col) for col in train.columns]
    valid.columns = [str(col) for col in valid.columns]
    test.columns = [str(col) for col in test.columns]
    return train, valid, test


def metrics(y, yhat):
    d = y - yhat
    mse_f = np.mean(d ** 2)
    mae_f = np.mean(abs(d))
    rmse_f = np.sqrt(mse_f)
    mape = np.mean(np.abs(yhat - y) / np.abs(y))
    mpe = np.mean((yhat - y) / y)
    corr = np.corrcoef(yhat, y)[0, 1]
    r2_f = 1 - (sum(d ** 2) / sum((y - np.mean(y)) ** 2))
    print("Results by manual calculation:\n",
          f'MAE: {mae_f:.4f} - MSE: {mse_f:.4f} - RMSE: {rmse_f:.4f} - R2: {r2_f:.4f}\n',
          f'MAPE: {mape:.4f} - MPE: {mpe:.4f} - CORR: {corr:.4f}')


def residuals(y, yhat):
    residual = y - yhat
    mean = residual.mean()
    median = residual.median()
    # skewness = 0 : normally distributed.
    # skewness > 0 : more weight in the left tail of the distribution. Long right tail. Median before mean.
    # skewness < 0 : more weight in the right tail of the distribution. Long left tail. Median after mean.
    skew = stats.skew(residual)
    sn.set()
    sn.displot(residual, kde='True')
    print(f'Residual information: \nMean: {mean} - Median: {median} - Skewness: {skew}')
    plt.show()


def check_stationarity(time_series):
    # Perform Dickey-Fuller test:
    # The null hypothesis (p-value > 0.05) for this test is that the data is not stationary.
    print('Results of Dickey-Fuller Test:')
    df_test = adfuller(time_series, autolag='AIC')
    df_output = Series(df_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in Series(df_test)[4].items():
        df_output[f'Critical Value (%{key:s})'] = value
    print(df_output)


def check_normal_distribution(data):
    print(f'Shapiro test: {stats.shapiro(data)}')
    print(f'Anderson test: {stats.normaltest(data)}')
    sn.displot(data=data, kde=True)
    plt.show()
    stats.probplot(data, dist="norm", plot=pylab)
    pylab.show()


def ohlc_chart(data, start=starting_date, end=ending_date, candle_size='W', volume=False):
    # Change the names of the columns.
    data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '5. adjusted close': 'Close'})
    data = data[start:end].resample(candle_size).mean()
    # Plot candlestick.
    new_style = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'axes.grid': True})
    mpf.plot(data, type='candle', ylabel='Price ($)', volume=volume, mav=(5, 10), figsize=(30, 7), style=new_style,
             scale_padding=0.05, xrotation=0, savefig=dict(fname="ohlc.png", bbox_inches="tight"))
    mpf.show()


def decompose_series(series, period=None, mode='multiplicative', fig_width=15):
    sn.set()
    result = seasonal_decompose(series, model=mode, period=period)
    result.plot().set_figwidth(fig_width)
    print(f'{mode}: [mean: {result.seasonal.mean()}, max:{result.seasonal.max()}, min:{result.seasonal.min()}]')
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

