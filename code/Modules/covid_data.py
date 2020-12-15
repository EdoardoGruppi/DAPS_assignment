# Import packages
from covid19dh import covid19
from pandas import DataFrame, to_pickle
import os
from Modules.config import *


def get_covid_data():
    dataframe, src = covid19(raw=True)
    dataframe_path = os.path.join(base_dir, 'Covid.pkl')
    to_pickle(dataframe, dataframe_path)
    return dataframe_path

