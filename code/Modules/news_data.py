# Import packages
from Modules.twitter_data import get_tweets, set_query
from pandas import read_pickle, concat, to_pickle


# Free News APIs do not allow to gather the necessary amount of data. The other solution is web scraping but it is not
# always legit and could lead a some sort of ban from the scraped site (e.g. google). The solution adopted consists in
# leveraging again the twitter scraper for instance to collect article headlines concerning the company from the main
# news provider. The twitter scraper already adopted enables also to retrieve information that are 3+ years old.
def get_news_tweeted(usernames, filename='News_twitted', words='', lang='en', since=None, until=None, mention=None,
                     hashtag=None):
    """
    Gets tweets sent by a particular user or a list of users and accordingly to the parameters passed. Then, it stores
    the data gathered in a pkl file.

    :param usernames: specify the users (i.e. sources) from which the tweets gathered are sent.
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
    # List of dataframes that will be captured fr each source selected
    dataframes = []
    for name in usernames:
        print(f'Scraping Twitter posts published by {name}...')
        # Create the correct query to search the tweets required
        query = set_query(words=words, username=name, lang=lang, until=until, since=since, mention=mention,
                          hashtag=hashtag)
        # Obtain the tweets sent by the current source considered
        dataframe_path = get_tweets(query, filename=filename)
        # Append the data gathered from each source
        dataframes.append(read_pickle(dataframe_path))
    # Create a single dataset concatenating all the dataframes collected
    dataframe = concat(dataframes)
    # If one or more datasets are retrieved save the unified dataset in a pickle file
    if dataframe_path is not None:
        to_pickle(dataframe, dataframe_path)
    return dataframe_path
