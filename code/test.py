# Import packages
from Modules.config import *
import os
from Modules.twitter_data import tweet_preprocessing
from Modules.stock_data import time_series_preprocessing
from Modules.utilities import ohlc_chart, plot_dataframe, transform_dataset, dataset_division
from pandas import to_pickle, DataFrame, read_pickle, concat
from Models.prophet import prophet_predictions
from Models.arima import arima_predictions

# STOCK DATA ACQUISITION AND STORAGE ===================================================================================
# If the datasets are not directly provided from the beginning there are two possibilities:
# 1. run the data_gatherer.py from the terminal
# 2. download the necessary information from the dedicated cloud database.
# todo Before processing use cloud database vedi oracle

# Once the datasets are locally available load the dataframes related to every pkl file.
time_series_dir = os.path.join(base_dir, 'Time_series.pkl')
# indicators_dir = os.path.join(base_dir, 'Indicators.pkl')
# indicators = read_pickle(indicators_dir)
# tweets_dir = os.path.join(base_dir, 'MSFT_twitter.pkl')
# news_dir = os.path.join(base_dir, 'News_twitted.pkl')

# DATA PREPROCESSING ===================================================================================================
time_series = time_series_preprocessing(df_path=time_series_dir)
# ohlc_chart(data=time_series, candle_size='10D', start=starting_date, end=ending_date, volume=False)
# news_dir = tweet_preprocessing(df_path=news_dir, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0)
# tweets_dir = tweet_preprocessing(df_path=tweets_dir, analysis='vader', like_weight=0, reply_weight=0,retweet_weight=0)

# remove -- tweets_dir = os.path.join(base_dir, f'MSFT_twitter_vader.pkl')
# tweets_dir = os.path.join(base_dir, f'MSFT_twitter_vader.pkl')
# tweets = read_pickle(tweets_dir)
# dataframe = time_series.join(tweets, how='left')

dataframe = time_series
train, valid, test = dataset_division(dataframe)
# train, valid, test = transform_dataset(train, valid, test, algorithm='pca', n_components=1, kernel='rbf',perplexity=5)

# DATA EXPLORATION =====================================================================================================

# DATA INFERENCE =======================================================================================================
# prophet_predictions(train, valid, regressor=True, mode='multiplicative')
arima_predictions(train, valid, regressor=True, mode='multiplicative')
# todo join test + valid or train + valid
# prophet_predictions(train, test, validation=False, mode='multiplicative')
# arima_predictions(train, test, regressor=True, mode='multiplicative')
