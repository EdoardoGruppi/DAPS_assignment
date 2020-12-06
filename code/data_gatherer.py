# Import packages
from Modules.stock_data import get_daily_time_series, get_indicator

# This file is used only to gather data in case the dataset can be directly provided.
# STOCK DATA ACQUISITION ===============================================================================================
time_series_dir = get_daily_time_series()
bbands_dir = get_indicator('bbands')
sma_dir = get_indicator('sma')

