# Import packages
from Modules.config import *
from alpha_vantage.timeseries import TimeSeries
from Modules.data_acquisition import get_daily_time_series, get_bbands
from alpha_vantage.sectorperformance import SectorPerformances
import matplotlib.pyplot as plt
import seaborn as sn
from Modules.visualization import plot_column
import pandas as pd
import os

# Cosa sono foreign exchange e cryptocurrencies
# cosa sono band e poi?
# cosa è sectorperformances
# bastano questi dati?


# time_series_dir = get_daily_time_series()
time_series_dir = os.path.join(base_dir, 'time_series.pkl')
time_series = pd.read_pickle(time_series_dir)
# bbands_dir = get_bbands(time_period=20, plot=True)
bbands_dir = os.path.join(base_dir, 'bbands.pkl')
bbands = pd.read_pickle(bbands_dir)

plot_column(time_series, '1. open')

sp = SectorPerformances(key='YOUR_API_KEY', output_format='pandas')
data, meta_data2 = sp.get_sector()
data['Rank A: Real-Time Performance'].plot(kind='bar')
plt.title('Real Time Performance (%) per Sector')
plt.tight_layout()
plt.grid()
plt.show()

