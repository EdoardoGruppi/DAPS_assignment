# Import packages
from Modules.stock_data import get_daily_time_series, get_multiple_indicators
from Modules.config import *
from Modules.twitter_data import get_tweets, set_query
from Modules.news_data import get_news_tweeted
from Modules.covid_data import get_covid_data

# STOCK DATA ACQUISITION ===============================================================================================
# time_series_dir = get_daily_time_series()
# indicators = ['bbands', 'sma', 'ema', 'rsi', 'adx']
# indicators_dir = get_multiple_indicators(indicators)
# covid_dir = get_covid_data()


# query = set_query(hashtag=f'${company}', lang='en', until=ending_test_period, since=starting_date, links=True)
# tweets_path = get_tweets(query=query, hide=True, filename='MSFT_twitter')
#
# usernames = ['nytimes', 'FinancialTimes', 'eToro', 'WSJ', 'TheEconomist', 'CNBC', 'forbes', 'barronsonline',
#              'YahooFinance', 'MarketWatch', 'washingtonpost']
# news_path = get_news_tweeted(usernames, filename='News_twitted', words='Microsoft', until=ending_test_period,
#                              since=starting_date)
