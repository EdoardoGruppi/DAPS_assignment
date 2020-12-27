# Import packages
from covid19dh import covid19
from pandas import DataFrame, to_pickle, read_pickle
import os
from Modules.config import *
from datetime import datetime
import numpy as np


def get_covid_data():
    dataframe, src = covid19(raw=True)
    dataframe_path = os.path.join(base_dir, 'Covid.pkl')
    to_pickle(dataframe, dataframe_path)
    return dataframe_path


def covid_preprocessing(df_path, daily_change=True):
    # todo makes some controls before
    dataframe = read_pickle(df_path)
    print(dataframe.describe())
    dataframe = dataframe[['date', 'confirmed', 'recovered', 'deaths']]
    dataframe.index = dataframe['date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    dataframe = dataframe.drop(['date'], axis=1)
    dataframe = dataframe.groupby(level=0).sum()
    # Considering only the active cases
    dataframe['active'] = dataframe.confirmed - dataframe.recovered - dataframe.deaths
    dataframe = dataframe.drop(['confirmed', 'deaths', 'recovered'], axis=1)
    if daily_change:
        # Considering the daily movement percentage of the active cases
        dataframe['daily_change'] = dataframe.active.pct_change(periods=1).fillna(0)
        dataframe.daily_change = dataframe.daily_change.replace(np.inf, 1)
        dataframe = dataframe.daily_change
    return dataframe
