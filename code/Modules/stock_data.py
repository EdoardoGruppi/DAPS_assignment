# Import packages
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from Modules.utilities import *
from pandas import DataFrame, concat, to_pickle
import pandas_datareader as pdr
from datetime import date
from Modules.config import *


def get_daily_time_series(filename='Time_series'):
    """
    Gets the daily stock movements of the company and from the starting date selected in the config.py

    :param filename: name to assign when saving the file. default_value='Time_series'
    :return: the path of the directory in which the data are stored.
    """
    ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    # Get the date, daily open, high, low, close, split, dividend, adjusted close and volume
    data, meta_data = ts.get_daily_adjusted(company, outputsize='full')
    data = data.sort_index(ascending=True)
    # Change the names of the columns. It is also required to load data on mongo db
    data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Original Close',
                                '5. adjusted close': 'Close', '6. volume': 'Volume', '7. dividend amount': 'Dividend',
                                '8. split coefficient': 'Split'})
    # Retain only the entries from the starting date selected in config.py until now
    data = data.loc[starting_date:]
    data.reset_index(level=0, inplace=True)
    # Save the dataset acquired in a pickle file
    data_directory = os.path.join(base_dir, f'{filename}.pkl')
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
    data_directory = os.path.join(base_dir, f'{indicator}.pkl')
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


def get_indexes(filename='Indexes'):
    """
    Gets the daily movements of S&P 100 and 500 along with Dow Jones 30.

    :param filename: name to assign when saving the file. default_value='Indexes'
    :return: the path of the directory in which the data are stored.
    """
    # List of indexes to retrieve
    indexes = ['^OEX', '^GSPC', '^DJI']
    # Empty list of dataframes. Then there will be one for each index
    dataframes = []
    for index in indexes:
        df = pdr.get_data_yahoo(index, start=starting_date, end=date.today())['Adj Close']
        dataframes.append(DataFrame(data={index: df.values}, index=df.index))
    # Concatenate dataframes
    dataframe = concat(dataframes, axis=1)
    # Rename columns
    dataframe = dataframe.rename(columns={'^OEX': 'S&p 100', '^GSPC': 'S&p 500', '^DJI': 'Dow Jones'})
    # Reset index to retain the date of the measurement
    dataframe.reset_index(level=0, inplace=True)
    # Save the dataset acquired in a pickle file
    data_directory = os.path.join(base_dir, f'{filename}.pkl')
    to_pickle(dataframe, data_directory)
    return data_directory


def time_series_preprocessing(time_series, indexes, method='linear', cap=None, nan=False, path=True, multi=False):
    """
    Pre-processes the time series dropping and combining columns. It allows also to operate outliers and missing values.

    :param time_series: name or path of the file where the time series is saved according to the value of the boolean
        argument path.
    :param indexes: name or path of the file where the time series related to the indexes is saved according to the
        value of the boolean argument path.
    :param method: interpolation method. default_value='linear'
    :param cap: state if univariate outliers have to be modified. It can be 'z-score' or 'iqr' to express which outliers
        to change. default_value=None
    :param nan: important only if cap is True. If True the outliers are substituted by NaN, otherwise they are replaced
        by the maximum value not detected as outlier. default_value=False
    :param path: if True time_series is the path of the file where the. Else the name. default_value=True
    :param multi: if True it can detect multivariate outliers using algorithms such as isolation forest.
        default_value=False
    :return:
    """
    if path:
        time_series = read_pickle(time_series)
        indexes = read_pickle(indexes)
    time_series = time_series.set_index('date')
    indexes = indexes.set_index('Date')
    time_series = combine_dataset([time_series, indexes])
    # Discard some columns that are not necessary like close, dividend amounts and split. In particular, adjusted close
    # is usually used to estimate historical correlation and volatility of companies stocks. The adjusted
    # closing price analyses the stock's dividends, stock splits and new stock offerings to determine an adjusted value.
    # Plot relationships between the features. Almost perfectly correlated features may be represented by only one var.
    # todo -- multivariate_visualization(time_series.drop(['Dividend', 'Split', 'Original Close'], axis=1))
    # scatter_plot(time_series, ['Open', 'High'])
    time_series = time_series[['Close', 'Volume', 'S&p 100']]
    # Detect outliers
    # todo -- detect_univariate_outlier(time_series, cap=cap, nan=nan)
    if multi:
        # the next line allows to detect multivariate outliers using for instance isolation forest.
        outliers_index = detect_multivariate_outlier(time_series, clf='iforest')
    # Add the missing days that are not considered since the stock market is closed during weekends and holidays.
    # In this case asfreq() could also be used instead of resample() since it is adopted only to add non-working day
    # and the sampling frequency is already daily.
    time_series = time_series.asfreq('D')
    # Interpolate to substitute each NaN with a likely value
    time_series = time_series.interpolate(method=method)
    return time_series


# volatility = time_series.Close.rolling(window=10, min_periods=10).std().replace(0, np.nan).bfill()
# time_series.Close /= volatility
# decompose_series(time_series.Close)
#
# volatility = series.rolling(window=365, min_periods=1).std().replace(0, np.nan).bfill()
# ajeje = series.values / volatility
# residuals_properties(ajeje)
