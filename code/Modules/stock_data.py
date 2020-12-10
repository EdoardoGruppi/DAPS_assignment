# Import packages
from Modules.config import *
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import to_pickle, DataFrame, read_pickle
import os


def get_daily_time_series():
    """
    Gets the daily stock movements of the company and from the starting date selected in the config.py

    :return: the path of the directory in which the data are stored.
    """
    ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    # Return date, daily open, daily high, daily low, daily close, daily split/dividend-adjusted close, daily volume
    data, meta_data = ts.get_daily_adjusted(company, outputsize='full')
    data = data.sort_index(ascending=True)
    # Retain only the entries from the starting date selected in config.py until now
    data = data.loc[starting_date:]
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
    data_directory = os.path.join(base_dir, '{}.pkl'.format(indicator))
    function = indicator_dict[indicator]
    data, meta_data1 = function(symbol=company, interval='daily', time_period=time_period)
    data = data.sort_index(ascending=True)
    # Retain only the entries from the starting date selected in config.py until now
    data = data.loc[starting_date:]
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
