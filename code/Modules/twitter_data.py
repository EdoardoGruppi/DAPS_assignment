# Import packages
import twint
from Modules.config import *
import os
from Modules.utilities import csv2pickle
from pandas import read_pickle, to_pickle
from datetime import datetime, timedelta
import re
from Modules.sentiment_analysis import flair_analysis, vader_analysis


def set_query(words='', hashtag=None, mention=None, username=None, lang=None, until=None, since=None, replies=False,
              links=True, min_likes=None, min_retweets=None, min_replies=None):
    """
    Returns a query to search tweets according to the parameters passed. It is essential to make functioning the twint
    package since it has some bugs in the phase of the query constructions.

    :param words: words that have to be present in the text of the tweet. default_value=''
    :param hashtag: hashtags (#example) that have to be present in the tweet. Insert only the name of the hashtag.
        default_value=None
    :param mention: mentions (@example) that have to be present in the tweet. Insert only the name of subject mentioned.
        default_value=None
    :param username: username from which tweets are sent. default_value=None
    :param lang: language of the tweets to collect. default_value=None
    :param until: the upper bound of the period of time in which tweets are published. default_value=None
    :param since: the lower bound of the period of time in which tweets are published. default_value=None
    :param replies: boolean to retrieve the replies along with the original tweets. default_value=False
    :param links: boolean to retrieve also the tweets that have a link inside the text. default_value=True
    :param min_likes: minimum number of likes that a tweet must have to be collected. default_value=None
    :param min_retweets: minimum number of retweets that a tweet must have to be collected. default_value=None
    :param min_replies: minimum number of replies that a tweet must have to be collected. default_value=None
    :return: the query created accordingly to the parameters passed.
    """
    query = words
    if hashtag is not None:
        query += f' (#{hashtag})'
    if mention is not None:
        query += f' (@{mention})'
    if username is not None:
        query += f' from:{username}'
    if lang is not None:
        query += f' lang:{lang}'
    if until is not None:
        query += f' until:{until}'
    if since is not None:
        query += f' since:{since}'
    if replies is False:
        query += f' -filter:replies'
    if links is False:
        query += f' -filter:links'
    if min_likes is not None:
        query += f' min_faves:{min_likes}'
    if min_retweets is not None:
        query += f' min_retweets:{min_retweets}'
    if min_replies is not None:
        query += f' min_replies:{min_replies}'
    return query


def get_tweets(query, hide=True, store=True, filename='Tweets'):
    """
    Gets tweets that respect the requirements expressed in the query and in the other parameters. Then, it can store
    data in a pkl file.

    :param query: twitter query from which get tweets.
    :param hide: if True outputs are not visible in the console. default_value=True
    :param store: if True tweets are stored in a pkl file. default_value=True
    :param filename: name of the pkl file. default_value='Tweets'
    :return: the path of the directory in which the data are stored.
    """
    filename = filename + '.csv'
    c = twint.Config()
    c.Custom_query = query
    c.Hide_output = hide
    c.Store_csv = store
    c.Output = os.path.join(base_dir, filename)
    twint.run.Search(c)
    dataframe_path = csv2pickle(filename)
    return dataframe_path


def clean_tweets(tweet):
    # todo improve the cleaning, tweet con una sola parola
    # Sub method replaces anything matching.
    # Delete all the https (and http) links that appear within the text.
    tweet = re.sub(r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,\\])+)", '', tweet)
    # Replace mentions and hashtags related to Microsoft and delete all the others.
    tweet = re.sub(r"(?i)[@#]Microsoft", 'Microsoft', tweet)
    tweet = re.sub(r"(?i)[@#][a-z0-9_]+", '', tweet)
    # Same for cashtags.
    tweet = re.sub(r"(?i)\$Msft", 'Microsoft', tweet)
    tweet = re.sub(r"(?i)\$[a-z]+", '', tweet)
    # Reduce the whitespaces between two words to only one.
    tweet = re.sub(r"\s+", ' ', tweet)
    return tweet


def tweet_preprocessing(df_path, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0):
    function_dict = {'vader': vader_analysis, 'flair': flair_analysis}
    tweets = read_pickle(df_path)
    # drop() function is not used since the number of columns to keep is lower than the number of columns to delete.
    # The new dataframe simply overwrites the previous one.
    tweets = tweets[['date', 'time', 'tweet', 'replies_count', 'retweets_count', 'likes_count']]
    tweets['date'] = tweets['date'] + ' ' + tweets['time']
    tweets['date'] = tweets['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    tweets = tweets.drop('time', axis=1)
    # If the message is twitted after the 21:00 pm in London (16:00 pm in NewYork - closing time of the nasdaq market)
    # it can influence only the session that takes place the day after.
    tweets['date'] = tweets['date'].apply(lambda x: x + timedelta(days=1) if x.hour > 20 else x)
    # todo tweets = tweets.set_index('date').sort_index()
    # Clean the text of the tweet
    tweets['tweet'] = tweets['tweet'].apply(clean_tweets)
    # Apply the analysis function selected
    function = function_dict[analysis]
    tweets = function(tweets, like_weight, reply_weight, retweet_weight, target_column='tweet')
    name = df_path.split(os.sep)[-1].split('.')[0]
    tweets_dir = os.path.join(base_dir, f'{name}_{analysis}.pkl')
    to_pickle(tweets, tweets_dir)
    return tweets_dir
