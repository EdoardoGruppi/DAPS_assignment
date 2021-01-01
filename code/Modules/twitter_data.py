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
    # Name of the file in which tweets will be saved. Twint store records into a csv file by default
    filename = filename + '.csv'
    # Configure Twint
    c = twint.Config()
    c.Custom_query = query
    c.Hide_output = hide
    c.Store_csv = store
    c.Output = os.path.join(base_dir, filename)
    twint.run.Search(c)
    # Transform the csv file saved in a pickle file
    dataframe_path = csv2pickle(filename)
    return dataframe_path


def clean_tweets(tweet):
    """
    Cleans the text of the tweet passed.

    :param tweet: tweet to clean.
    :return: the cleaned tweet.
    """
    # Sub method replaces anything matching.
    # Delete all the https (and http) links that appear within the text.
    tweet = re.sub(r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*,\\])+)", '', tweet)
    # Replace mentions and hashtags related to Microsoft and delete all the others.
    tweet = re.sub(r"(?i)[@#]Microsoft", 'Microsoft', tweet)
    tweet = re.sub(r"(?i)[@#][a-z0-9_]+", '', tweet)
    # Same for cashtags.
    tweet = re.sub(r"(?i)\$Msft", 'Microsoft', tweet)
    tweet = re.sub(r"(?i)\$[a-z]+", '', tweet)
    # Reduce the whitespaces between two words to only one space.
    tweet = re.sub(r"\s+", ' ', tweet)
    return tweet


def tweet_preprocessing(df_path, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0, move=0):
    """
    Pre-processes the dataset of tweets by dropping and combining columns. It can consider tweets as they are published
    in a different day by means of the parameter move and it cleans the text of every message posted before applying
    sentiment analysis with one algorithm between two possibilities (vader and flair). Finally the preprocessed dataset
    :raise saved in a new dedicated pickle file.

    :param df_path: path of the original dataset containing tweets.
    :param analysis: analysis methodology to adopt, i.e. 'vader' or 'flair'. default_value='vader'
    :param like_weight: weight assigned to the post likes. If likes, replies and retweets weights are all 0 they are
        not considered. default_value=0
    :param reply_weight: weight assigned to the post replies. If likes, replies and retweets weights are all 0 they are
        not considered. default_value=0
    :param retweet_weight: weight assigned to the post retweets. If likes, replies and retweets weights are all 0 they
        are not considered. default_value=0
    :param move: indicates the London time (hours) after which tweets are moved to the next day. If 0 no tweet is
        shifted. default_value=0
    :return: the path to the file in which the new tweets dataset is saved.
    """
    # Dictionary of possible functions for sentiment analysis
    function_dict = {'vader': vader_analysis, 'flair': flair_analysis}
    # Read the dataset from the passed path
    tweets = read_pickle(df_path)
    # drop() function is not used since the number of columns to keep is lower than the number of columns to delete.
    # The new dataframe simply overwrites the previous one.
    tweets = tweets[['date', 'time', 'tweet', 'replies_count', 'retweets_count', 'likes_count']]
    # Date and time are combined and substituted by a datetime attribute. This allows to avoid discrepancies when
    # datasets will be joined together and facilitate several operations performed.
    tweets['date'] = tweets['date'] + ' ' + tweets['time']
    tweets['date'] = tweets['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    tweets = tweets.drop('time', axis=1)
    if move != 0:
        # 16:00 pm in NewYork (closing time of the nasdaq market corresponds) to 21:00 pm in London.
        # To consider the influence that a tweet has only in a particular session, e.g. that takes place the day after.
        tweets['date'] = tweets['date'].apply(lambda x: x + timedelta(days=1) if x.hour > move else x)
    # Clean the text of each tweet
    tweets['tweet'] = tweets['tweet'].apply(clean_tweets)
    # Select and apply the analysis function required
    function = function_dict[analysis]
    tweets = function(tweets, like_weight, reply_weight, retweet_weight, target_column='tweet')
    # Create a new filename starting from the one passed
    name = df_path.split(os.sep)[-1].split('.')[0]
    tweets_dir = os.path.join(base_dir, f'{name}_{analysis}.pkl')
    # Save the new dataset in a dedicated novel pickle file
    to_pickle(tweets, tweets_dir)
    return tweets_dir
