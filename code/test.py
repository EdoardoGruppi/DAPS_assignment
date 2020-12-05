# Import packages
from Modules.config import *
from Modules.data_acquisition import get_daily_time_series, get_indicator
from Modules.visualization import plot_dataframe
import pandas as pd
import os

# # time_series_dir = get_daily_time_series()
# # bbands_dir = get_indicator('bbands')
# # sma_dir = get_indicator('sma')
time_series_dir = os.path.join(base_dir, 'time_series.pkl')
bbands_dir = os.path.join(base_dir, 'bbands.pkl')
sma_dir = os.path.join(base_dir, 'sma.pkl')

time_series = pd.read_pickle(time_series_dir)
bbands = pd.read_pickle(bbands_dir)
sma = pd.read_pickle(sma_dir)

plot_dataframe(time_series, '4. close')
plot_dataframe(sma)


