# Import packages
from Modules.twitter_data import get_tweets, set_query
from pandas import read_pickle, concat, to_pickle


# Free News APIs are a little number and do not allow to gather the necessary amount of data. The other solution is
# web scraping but it is not really legit and could lead a some sort of ban from the site (e.g. google).
# The solution is to use again the twitter scraper to collect headlines of articles concerning Microsoft
# from the main news provider. Again, scraping is not good but in the case of the twitter scraper already adopted
# it works well and enables to retrieve information that are 3+ years old.
def get_news_tweeted(usernames, filename='News_twitted', words='', lang='en', since=None, until=None, mention=None,
                     hashtag=None):
    """
    Gets tweets sent by a particular user or a list of users and accordingly to the parameters passed. Then, it stores
    the data gathered in a pkl file.

    :param usernames: specify the users from which the tweets gathered are sent.
    :param filename: name of the pickle file in which data are saved.
    :param words: words that have to be present in the text of the tweet. default_value=''
    :param lang: language of the tweets to collect. default_value=None
    :param since: the lower bound of the period of time in which tweets are published. default_value=None
    :param until: the upper bound of the period of time in which tweets are published. default_value=None
    :param mention: mentions (@example) that have to be present in the tweet. Insert only the name of subject mentioned.
        default_value=None
    :param hashtag: hashtags (#example) that have to be present in the tweet. Insert only the name of the hashtag.
        default_value=None
    :return: the path of the directory in which the data are stored.
    """
    dataframe_path = None
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
