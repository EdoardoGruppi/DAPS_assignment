# Import packages
from Modules.twitter_data import tweet_preprocessing
from Modules.stock_data import time_series_preprocessing
from Modules.utilities import *
from pandas import concat
from Models.prophet import prophet_predictions
from Models.arima import arima_predictions, arima_test
from Modules.covid_data import covid_preprocessing
from Modules.exploration import *
from Modules.mongo_db import download_datasets

# STOCK DATA ACQUISITION AND STORAGE ===================================================================================
# If the datasets are not directly provided from the beginning there are two possibilities:
# 1. download the necessary information from the dedicated cloud database. The data was previously uploaded through the
#    upload_datasets() available within the module named mongo_db.
# 2. run the data_gatherer script from the terminal. However, this is not the preferred solution since the entire
#    process of data acquisition takes at least 70-80 minutes. Moreover, the code is based on the version of the
#    datasets downloaded from MongoDB. It varies a little because of the conversion imposed by the cloud database
#    platform, i.e. the datetime index is transformed in an attribute of the dataset.
# Consequently, the first option is adopted.
download_datasets()
# Once the datasets are locally available load the dataframes related to every pkl file.
indexes_dir = os.path.join(base_dir, 'Indexes.pkl')
time_series_dir = os.path.join(base_dir, 'Time_series.pkl')
covid_dir = os.path.join(base_dir, 'Covid.pkl')
tweets_dir = os.path.join(base_dir, 'MSFT_twitter.pkl')
news_dir = os.path.join(base_dir, 'News.pkl')

# DATA PREPROCESSING ===================================================================================================
stock_data = time_series_preprocessing(time_series_dir, indexes_dir, path=True)
# The Ohlc chart will be saved inside the code folder. See ohlc.png file.
ohlc_chart(path=time_series_dir, candle_size='10D', start=starting_date, end=ending_date, volume=False)
tweets = tweet_preprocessing(tweets_dir, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0, move=21)
news = tweet_preprocessing(news_dir, analysis='flair', like_weight=0, reply_weight=0, retweet_weight=0, move=21)
covid = covid_preprocessing(covid_dir, daily_change=True)
# Change the column names that can create conflicts
news.name = 'Mood'
# Create a unified dataset
dataframe = combine_dataset([stock_data, tweets, news, covid])
train, valid, test = dataset_division(dataframe)
del time_series_dir, covid_dir, news_dir, tweets_dir, indexes_dir

# DATA EXPLORATION AND HYPOTHESIS TESTING ==============================================================================
dataframe, new_dataframe, columns = change_format(concat([train, valid]))
# Search seasonality in the data
decompose_series(dataframe.Close, mode='multiplicative')
attributes_visualization(new_dataframe, columns, hue=['Day', 'Month', 'Year', 'Quarter', 'WeekDay'])
plot_auto_correlation(dataframe.Close, partial=False, lags=365)
# Visualize attribute relationships
attributes_visualization(new_dataframe, columns, hue=None)
multivariate_visualization(dataframe)
scatter_plot(dataframe, ['Mood', 'Close'])
check_stationarity(dataframe['Close'])

dataframe_pct = percentage_change(dataframe, ['Covid'])
# Independence of the observations so that there is no relationship between the observations in each group.
plot_auto_correlation(dataframe_pct.Close)
print('\nHypothesis Testing...')
custom_test_1(dataframe_pct.Close, dataframe_pct.Sentiment, threshold=0.10, significance_level=0.06, test=1)
custom_test_2(dataframe_pct.Close, dataframe.Volume, percentile=65, test=2)
custom_test_2(dataframe_pct.Close, dataframe_pct.Volume, percentile=10, test=2)
custom_test_1(dataframe_pct.Sentiment, dataframe.Covid, threshold=0.35, significance_level=0.06, test=2)
custom_test_2(dataframe_pct.Close, dataframe_pct['S&p 100'], percentile=0.10, test=1)
custom_test_1(dataframe_pct.Volume, dataframe.Covid, threshold=0.90, significance_level=0.05, test=2)
custom_test_2(dataframe_pct.Volume, dataframe_pct.Sentiment, percentile=50, test=1)
custom_test_2(dataframe_pct.Sentiment, dataframe.Volume, percentile=40, significance_level=0.07, test=2)
custom_test_2(dataframe.Covid, dataframe_pct.Sentiment, percentile=90, test=0)
custom_test_2(dataframe_pct.Close, dataframe.Sentiment, percentile=50, test=1)

del new_dataframe, dataframe, columns
train = train.drop(['Mood'], axis=1)
valid = valid.drop(['Mood'], axis=1)
test = test.drop(['Mood'], axis=1)
train_stock, valid_stock, test_stock = dataset_division(stock_data.drop(['S&p 100'], axis=1))
train_stock, valid_stock, test_stock = transform_dataset(train_stock, valid_stock, test_stock, reduction=False)
train, valid, test = transform_dataset(train, valid, test, algorithm='pca', n_components=0.90, reduction=True)

# DATA INFERENCE =======================================================================================================
# Selection of the hyper-parameters
# The validation split is used to set all the hyper-parameters that cannot be found with grid search algorithms
# prophet_predictions(train_stock, valid_stock, regressor=True, mode='multiplicative', holidays=False)
# arima = arima_predictions(train_stock, valid_stock, regressor=True)
# Compare models using only company's stock data
train_stock = concat([train_stock, valid_stock])
prophet_predictions(train_stock, test_stock, regressor=True, mode='multiplicative', holidays=False)
arima = arima_predictions(train_stock, test_stock, regressor=True)
# Re-execute the prediction using the model that led to best results in the previous step
train = concat([train, valid])
arima_test(model=arima, train=train, test=test, regressor=True)
