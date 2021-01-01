# Import packages
from Modules.config import *
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
from pandas import to_pickle, DataFrame, read_pickle
import os
import seaborn as sn
from Modules.utilities import detect_univariate_outlier, multivariate_visualization, detect_multivariate_outlier


def get_daily_time_series():
    """
    Gets the daily stock movements of the company and from the starting date selected in the config.py

    :return: the path of the directory in which the data are stored.
    """
    ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    # Get the date, daily open, high, low, close, split, dividend, adjusted close and volume
    data, meta_data = ts.get_daily_adjusted(company, outputsize='full')
    data = data.sort_index(ascending=True)
    # Retain only the entries from the starting date selected in config.py until now
    data = data.loc[starting_date:]
    # Save the dataset acquired in a pickle file
    data_directory = os.path.join(base_dir, 'Time_series.pkl')
    to_pickle(data, data_directory)
    return data_directory


def get_indicator(indicator, time_period=20, plot=True):
    """
    Gets a particular indicator associated to the company and from the starting date selected in the file config.py.

    :param indicator: specifies which indicator to retrieve.
    :param time_period: number of data points used to calculate the indicator values. default_value=20
    :param plot: if True it plots the data inside the dataframe downloaded. default_value=True
    :return: the path of the directory in which the data are stored.
    """
    ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    indicator_dict = {'bbands': ti.get_bbands, 'sma': ti.get_sma, 'ema': ti.get_ema, 'rsi': ti.get_rsi,
                      'adx': ti.get_adx, 'cci': ti.get_cci, 'aroon': ti.get_aroon}
    function = indicator_dict[indicator]
    data, meta_data1 = function(symbol=company, interval='daily', time_period=time_period)
    data = data.sort_index(ascending=True)
    # Retain only the entries from the starting date selected in config.py until now
    data = data.loc[starting_date:]
    # Save the dataset acquired in a pickle file
    data_directory = os.path.join(base_dir, '{}.pkl'.format(indicator))
    to_pickle(data, data_directory)
    if plot:
        sn.set()
        data.plot()
        plt.title(f'{indicator} indicator for {company} stock')
        plt.show()
    return data_directory


def get_multiple_indicators(indicators, time_period=20):
    """
    Gets a particular indicator associated to the company and from the starting date selected in the file config.py.

    :param indicators: list of the indicators to retrieve.
    :param time_period: number of data points used to calculate the indicator values. default_value=20
    :return: the path of the directory in which the data are stored.
    """
    ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    indicator_dict = {'bbands': ti.get_bbands, 'sma': ti.get_sma, 'ema': ti.get_ema, 'rsi': ti.get_rsi,
                      'adx': ti.get_adx, 'cci': ti.get_cci, 'aroon': ti.get_aroon}
    data_directory = os.path.join(base_dir, 'Indicators.pkl')
    dataframe = DataFrame()
    for indicator in indicators:
        function = indicator_dict[indicator]
        data, meta_data1 = function(symbol=company, interval='daily', time_period=time_period)
        data = data.sort_index(ascending=True)
        # Retain only the entries from the starting date selected in config.py until now
        data = data.loc[starting_date:]
        dataframe = dataframe.join(data, how='outer')
    to_pickle(dataframe, data_directory)
    return data_directory


def time_series_preprocessing(time_series, method='linear', cap=None, nan=False, path=True):
    """
    Pre-processes the time series dropping and combining columns. It allows also to operate outliers and missing values.

    :param time_series: name or path of the file where the time series is saved according to the value of the boolean
        argument path.
    :param method: interpolation method. default_value='linear'
    :param cap: state if univariate outliers have to be modified. It can be 'z-score' or 'iqr' to express which outliers
        to change. default_value=None
    :param nan: important only if cap is True. If True the outliers are substituted by NaN, otherwise they are replaced
        by the maximum value not detected as outlier. default_value=False
    :param path: if True time_series is the path of the file where the. Else the name. default_value=True
    :return:
    """
    if path:
        time_series = read_pickle(time_series)
    # Discard some columns that are not necessary like close, dividend amounts and split. In particular, adjusted close
    # is usually used to estimate historical correlation and volatility of companies stocks. The adjusted
    # closing price analyses the stock's dividends, stock splits and new stock offerings to determine an adjusted value.
    # Plot relationships between the features. Almost perfectly correlated features may be represented by only one var.
    # todo -- multivariate_visualization(time_series.drop(['7. dividend amount', '8. split coefficient'], axis=1))
    time_series = time_series[['5. adjusted close', '6. volume']]
    # Change name columns. This will be useful if plots are required.
    time_series = time_series.rename(columns={'5. adjusted close': 'Close', '6. volume': 'Volume'})
    # Detect outliers
    # todo -- detect_univariate_outlier(time_series, cap=cap, nan=nan)
    # the next line in case you want detect multivariate outliers using for instance isolation forest.
    # outliers_index = detect_multivariate_outlier(time_series, clf='iforest')
    # Add the missing days that are not considered since the stock market is closed during weekends and holidays.
    # In this case asfreq() could also be used instead of resample() since it is adopted only to add non-working day
    # and the sampling frequency is already daily.
    time_series = time_series.asfreq('D')
    # Interpolate to substitute each NaN with a likely value
    time_series = time_series.interpolate(method=method)
    return time_series
