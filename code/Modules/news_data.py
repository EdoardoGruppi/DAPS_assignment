# Import packages
from Modules.twitter_data import get_tweets, set_query
from pandas import read_pickle, concat, to_pickle


# Free News APIs are a little number and do not allow to gather the necessary amount of data. The other solution is
# web scraping but it is not really legit and couldlead a some sort of ban from the site (e.g. google).
# The solution is to use again the twitter scraper to collect headlines of articles concerning Microsoft
# from the main news provider. Again, scraping is not good but in the case of the twitter scraper already adopted
# it works well and enables to retrieve information that are even 3 years old.
def get_news_tweeted(usernames, filename='News_twitted', dataframe_path=None, words='', lang='en', since=None,
                     until=None, mention=None, hashtag=None):
    dataframes = []
    for name in usernames:
        print(f'{name}')
        query = set_query(words=words, username=name, lang=lang, until=until, since=since, mention=mention,
                          hashtag=None)
        dataframe_path = get_tweets(query, filename=filename)
        dataframes.append(read_pickle(dataframe_path))
    dataframe = concat(dataframes)
    if dataframe_path is not None:
        to_pickle(dataframe, dataframe_path)
    return dataframe_path
