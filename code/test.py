# Import packages
from Modules.config import *
from Modules.utilities import plot_dataframe
import pandas as pd
import os

# STOCK DATA ACQUISITION ===============================================================================================
time_series_dir = os.path.join(base_dir, 'time_series.pkl')
bbands_dir = os.path.join(base_dir, 'bbands.pkl')
sma_dir = os.path.join(base_dir, 'sma.pkl')

time_series = pd.read_pickle(time_series_dir)
bbands = pd.read_pickle(bbands_dir)
sma = pd.read_pickle(sma_dir)

plot_dataframe(time_series, '4. close')
plot_dataframe(sma)


