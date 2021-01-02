# Import packages
from Modules.config import *
import os
from Modules.twitter_data import tweet_preprocessing
from Modules.stock_data import time_series_preprocessing
from Modules.utilities import *
from pandas import to_pickle, DataFrame, read_pickle, concat
from Models.prophet import prophet_predictions
from Models.arima import arima_predictions, arima_test
from Modules.covid_data import covid_preprocessing
from Modules.exploration import *

# STOCK DATA ACQUISITION AND STORAGE ===================================================================================
# If the datasets are not directly provided from the beginning there are two possibilities:
# 1. run the data_gatherer.py from the terminal
# todo -- 2. download the necessary information from the dedicated cloud database.

# Once the datasets are locally available load the dataframes related to every pkl file.
time_series_dir = os.path.join(base_dir, 'Time_series.pkl')
covid_dir = os.path.join(base_dir, 'Covid.pkl')
# tweets_dir = os.path.join(base_dir, 'MSFT_twitter.pkl')
# news_dir = os.path.join(base_dir, 'News.pkl')

# DELETE LINES BELOW ======================
tweets_dir = os.path.join(base_dir, f'MSFT_twitter_vader.pkl')
tweets = read_pickle(tweets_dir)
news_dir = os.path.join(base_dir, 'News_flair.pkl')
news = read_pickle(news_dir)

# DATA PREPROCESSING ===================================================================================================
time_series = time_series_preprocessing(time_series=time_series_dir, path=True)
# ohlc_chart(data=time_series, candle_size='10D', start=starting_date, end=ending_date, volume=False)
# tweets = tweet_preprocessing(df_path=tweets_dir, analysis='vader', like_weight=4, reply_weight=1, retweet_weight=8)
# news = tweet_preprocessing(df_path=news_dir, analysis='flair', like_weight=4, reply_weight=1, retweet_weight=8)
covid = covid_preprocessing(covid_dir, daily_change=True)
# Change the column names that can create conflicts
news.name = 'Mood'
# Create a unified dataset
dataframe = combine_dataset([time_series, covid, tweets, news])
dataframe = shift_dataset(dataframe)
train, valid, test = dataset_division(dataframe)
del time_series_dir, covid_dir, news_dir, tweets_dir

# DATA EXPLORATION =====================================================================================================
dataframe, new_dataframe, columns = change_format(concat([train, valid]))
# # multivariate_visualization(dataframe)
# # attributes_visualization(new_dataframe, columns, hue=['Day', 'Month', 'Year', 'Quarter', 'WeekDay'])
# # plot_rolling(dataframe['Close'], window=7)
# decompose_series(dataframe['Close'], mode='multiplicative')
# check_stationarity(dataframe['Close'])
granger_test(dataframe, 'Close')
# del new_dataframe, dataframe, columns

train_stock, valid_stock, test_stock = dataset_division(shift_dataset(time_series))
train_stock, valid_stock, test_stock = transform_dataset(train_stock, valid_stock, test_stock, reduction=False)
train, valid, test = transform_dataset(train, valid, test, algorithm='pca', n_components=0.90, reduction=True)

# DATA INFERENCE =======================================================================================================
# Validate and compare models using only company's stock data
prophet_predictions(train_stock, valid_stock, regressor=True, mode='multiplicative', holidays=False)
# arima = arima_predictions(train_stock, valid_stock, regressor=True)
# The validation split is used to set all the hyper-parameters that cannot be found with grid search algorithms
train_stock = concat([train_stock, valid_stock])
prophet_predictions(train_stock, test_stock, regressor=True, mode='multiplicative', holidays=False)
# arima_test(model=arima, train=train_stock, test=test_stock, regressor=True)
# todo delete one of the next lines
# Remake the prediction using the model that led to best results in the previous step
train = concat([train, valid])
prophet_predictions(train, test, regressor=True, mode='multiplicative', holidays=False)
