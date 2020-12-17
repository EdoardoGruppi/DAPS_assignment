# Import packages
from covid19dh import covid19
from pandas import DataFrame, to_pickle, read_pickle
import os
from Modules.config import *
from datetime import datetime


def get_covid_data():
    dataframe, src = covid19(raw=True)
    dataframe_path = os.path.join(base_dir, 'Covid.pkl')
    to_pickle(dataframe, dataframe_path)
    return dataframe_path


def covid_preprocessing(df_path):
    # todo makes some controls before
    dataframe = read_pickle(df_path)
    print(dataframe.describe())
    dataframe = dataframe[['date', 'confirmed', 'recovered', 'deaths']]
    dataframe.index = dataframe['date'].apply(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    dataframe = dataframe.drop(['date'], axis=1)
    dataframe = dataframe.groupby(level=0).sum()
    return dataframe
