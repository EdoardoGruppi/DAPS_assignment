# Import packages
from Modules.stock_data import get_daily_time_series, get_indicator
import os
from Modules.config import *
from Modules.twitter_data import get_tweets, set_query
from Modules.news_data import get_news_tweeted
import pandas as pd

# This file is used only to gather data in case the dataset cannot be directly provided.
# STOCK DATA ACQUISITION ===============================================================================================
# time_series_dir = get_daily_time_series()
# bbands_dir = get_indicator('bbands')
# sma_dir = get_indicator('sma')

# query = set_query(hashtag=f'${company}', lang='en', until='2020-05-01', since='2017-04-01', links=True)
# tweets_path = get_tweets(query=query, hide=True, filename='MSFT_twitter')

# usernames = ['nytimes', 'FinancialTimes', 'eToro', 'WSJ', 'TheEconomist', 'CNBC', 'forbes', 'barronsonline',
#              'YahooFinance', 'MarketWatch', 'washingtonpost']
# news_path = get_news_tweeted(usernames, filename='News_twitted', words='Microsoft', until='2020-05-01',
#                              since='2017-04-01')

# query = set_query(username='irrisolvibile', lang='it', until='2020-12-10', since='2020-12-08', links=True)
# tweets_path = get_tweets(query=query, hide=True, filename='a')

# Before processing use cloud database
