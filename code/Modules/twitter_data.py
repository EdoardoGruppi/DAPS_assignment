# Import packages
import twint
from Modules.config import *
import os
from Modules.utilities import csv2pickle


def set_query(words='', hashtag=None, mention=None, username=None, lang=None, until=None, since=None, replies=False,
              links=True, min_likes=None, min_retweets=None, min_replies=None):
    """
    Returns a query to search tweets according to the parameters passed. It is essential to make functioning the twint
    package since it has some bugs in the phase of the query constructions.

    :param words:
    :param hashtag:
    :param mention:
    :param username:
    :param lang:
    :param until:
    :param since:
    :param replies:
    :param links:
    :param min_likes:
    :param min_retweets:
    :param min_replies:
    :return:
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
    Get tweets that respect the requirements expressed in the query and in the other parameters. Then, it can store
    data in a pkl file.

    :param query: twitter query from which get tweets.
    :param hide: if True outputs are not visible in the console. default_value=True
    :param store: if True tweets are stored in a pkl file. default_value=True
    :param filename: name of the pkl file. default_value='Tweets'
    :return: the directory in which the data are stored.
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
