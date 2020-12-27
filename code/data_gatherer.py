# Import packages
from Modules.stock_data import get_daily_time_series, get_multiple_indicators
from Modules.config import *
from Modules.twitter_data import get_tweets, set_query
from Modules.news_data import get_news_tweeted
from Modules.covid_data import get_covid_data

# STOCK DATA ACQUISITION ===============================================================================================
time_series_dir = get_daily_time_series()
# indicators = ['bbands', 'sma', 'ema', 'rsi', 'adx']
# indicators_dir = get_multiple_indicators(indicators)
covid_dir = get_covid_data()


query = set_query(hashtag=f'${company}', lang='en', until=ending_test_period, since=starting_date, links=True)
tweets_path = get_tweets(query=query, hide=True, filename='MSFT_twitter')

# Newspaper, financial blogger, economists, tv content provider
usernames = ['nytimes', 'FinancialTimes', 'eToro', 'WSJ', 'TheEconomist', 'CNBC', 'forbes', 'barronsonline',
             'YahooFinance', 'MarketWatch', 'washingtonpost', 'latimes', 'USATODAY', 'nypost', 'time', 'Gizmodo',
             'clevelanddotcom', 'chicagotribune', 'denverpost', 'ajc', 'M_McDonough', 'TechCrunch',
             'statesman', 'seattletimes', 'guardian', 'freep', 'NYDailyNews', 'Suntimes', 'cnnbrk', 'CNN', 'cnni',
             'SFGate', 'HoustonChron', 'azcentral', 'PhillyInquirer', 'Oregonian', 'StarTribune', 'FoxNews', 'verge',
             'RollingStone', 'guardiantech', 'observer', 'thenextweb', 'tech2eets', 'DigitalTrends', 'arstechnica',
             'LasVegasSun', 'sdut', 'reviewjournal', 'StarAdvertiser', 'BBCBreaking', 'MSNBC', 'mashable',
             'WIRED', 'GoldmanSachs', 'ritholtz', 'SkyNews', 'SkyNewsBiz', 'AJENews', 'euronews', 'smashingmag',
             'DavidSchawel', 'howardlindzon', 'conorsen', 'mark_dow', 'cullenroche', 'nanexllc', 'HarvardBiz',
             'michaelkitces', 'firoozye', 'elerianm', 'valuewalk', 'jasonzweigwsj']
# Control if any source is present twice in the list above
usernames = list(set(usernames))
news_path = get_news_tweeted(usernames, filename='News', words=company_extended, until=ending_test_period,
                             since=starting_date)
