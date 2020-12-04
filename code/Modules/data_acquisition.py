# Import packages
from Modules.config import *
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os


def get_daily_time_series():
    ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    # Return date, daily open, daily high, daily low, daily close, daily split/dividend-adjusted close, daily volume
    data, meta_data = ts.get_daily_adjusted(company, outputsize='full')
    # Retain only the entries from April 2017
    data = data.loc[:'2017-04-01 00:00:00']
    data_directory = os.path.join(base_dir, 'time_series.pkl')
    pd.to_pickle(data, data_directory)
    return data_directory


def get_bbands(time_period=20, plot=True):
    ti = TechIndicators(key=alpha_vantage_api_key, output_format='pandas', indexing_type='date')
    data, meta_data1 = ti.get_bbands(symbol=company, interval='daily', time_period=time_period)
    # Retain only the entries from April 2017
    data = data.loc[:'2017-04-01 00:00:00']
    data_directory = os.path.join(base_dir, 'bbands.pkl')
    pd.to_pickle(data, data_directory)
    if plot:
        sn.set()
        data.plot()
        plt.title('BBbands indicator for Microsoft stock')
        plt.show()
    return data_directory



