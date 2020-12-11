# Import packages
from Modules.config import *
from pandas import read_pickle
import os
from Modules.twitter_data import tweet_preprocessing
# from Modules.stock_data import time_series_preprocessing


# STOCK DATA ACQUISITION ===============================================================================================
# If the datasets are not directly provided from the beginning there are two possibilities:
# 1. run the data_gatherer.py from the terminal
# 2. download the necessary information from the dedicated cloud database.
# todo Before processing use cloud database vedi oracle

# Once the datasets are locally available load the dataframes related to every pkl file.
# todo so far indicators are not considered
time_series_dir = os.path.join(base_dir, 'Time_series.pkl')
tweets_dir = os.path.join(base_dir, 'MSFT_twitter.pkl')
news_dir = os.path.join(base_dir, 'News_twitted.pkl')

# time_series = read_pickle(time_series_dir)
# tweets = read_pickle(tweets_dir)
# news = read_pickle(news_dir)

# DATA PREPROCESSING ===================================================================================================

# Now that you have the data stored, you can start preprocessing it. Think about what features to
# keep, which ones to transform, combine or discard. Make sure your data is clean and consistent
# (e.g., are there many outliers? any missing values?). You are expected to (1) clean, (2) visualize
# and (3) transform your data (e.g., using normalization, dimensionality reduction, etc.).

# time_series_preprocessing(time_series_dir)
tweets = tweet_preprocessing(tweets_dir, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0)
