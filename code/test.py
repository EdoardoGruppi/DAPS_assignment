# Import packages
from Modules.config import *
import os
from Modules.twitter_data import tweet_preprocessing
from Modules.stock_data import time_series_preprocessing
from Modules.utilities import *
from pandas import to_pickle, DataFrame, read_pickle, concat
from Models.prophet import prophet_predictions
from Models.arima import arima_predictions
from Modules.covid_data import covid_preprocessing

# STOCK DATA ACQUISITION AND STORAGE ===================================================================================
# If the datasets are not directly provided from the beginning there are two possibilities:
# 1. run the data_gatherer.py from the terminal
# 2. download the necessary information from the dedicated cloud database.
# todo Before processing use cloud database vedi oracle

# Once the datasets are locally available load the dataframes related to every pkl file.
time_series_dir = os.path.join(base_dir, 'Time_series.pkl')
# indicators_dir = os.path.join(base_dir, 'Indicators.pkl')
covid_dir = os.path.join(base_dir, 'Covid.pkl')
# tweets_dir = os.path.join(base_dir, 'MSFT_twitter.pkl')
# news_dir = os.path.join(base_dir, 'News_twitted.pkl')
# indicators = read_pickle(indicators_dir)
# time_series = read_pickle(time_series_dir)

# DATA PREPROCESSING ===================================================================================================
time_series = time_series_preprocessing(time_series=time_series_dir, path=True)
# ohlc_chart(data=time_series, candle_size='10D', start=starting_date, end=ending_date, volume=False)
# news_dir = tweet_preprocessing(df_path=news_dir, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0)
# tweets_dir = tweet_preprocessing(df_path=tweets_dir, analysis='vader', like_weight=0, reply_weight=0,retweet_weight=0)
covid = covid_preprocessing(covid_dir)

# tweets_dir = os.path.join(base_dir, f'MSFT_twitter_vader.pkl')
# tweets = read_pickle(tweets_dir)

detect_seasonality(time_series, 'Close')

dataframe = combine_dataset([time_series, covid])
# granger_test(dataframe, ['Close', 'Volume'])
dataframe = shift_dataset(dataframe)
train, valid, test = dataset_division(dataframe)
train, valid, test = transform_dataset(train, valid, test, algorithm='pca', n_components=1, kernel='rbf',
                                       reduction=False)

# DATA EXPLORATION =====================================================================================================

# DATA INFERENCE =======================================================================================================
prophet_predictions(train, valid, regressor=True, mode='multiplicative', holidays=False)
# arima_predictions(train, valid, regressor=True)
# The validation split is used to set all the hyper-parameters that cannot be found with grid search algorithms
# todo control
# train, valid, test = dataset_division(dataframe, valid_size=0)
# train, valid, test = transform_dataset(train, valid, test, algorithm='pca', n_components=1, kernel='rbf',
#                                        reduction=False)
# prophet_predictions(train, test, validation=False, mode='multiplicative')
# arima_predictions(train, test, regressor=True)
