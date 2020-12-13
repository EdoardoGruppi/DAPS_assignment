# Import Packages
import flair
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download


def flair_analysis(tweets, like_weight, reply_weight, retweet_weight, target_column='tweet'):
    """
    Performs natural language processing on the text of the tweets passed by means of Flair. Then it returns the tweets
    with the associated results of the sentiment analysis.

    :param tweets: dataframe that contains the tweets.
    :param like_weight: influence of each like on the related tweet.
    :param reply_weight: influence of each reply on the related tweet.
    :param retweet_weight: influence of each retweet on the related tweet.
    :param target_column: column of the dataframe that contains the text of the tweets. default_value='tweet'
    :return: the tweets grouped by day with the average sentiment score.
    """
    results = []
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    print('Flair analysis started...')
    for count, tweet in enumerate(tweets[target_column].to_list(), 1):
        if count % 500 == 0:
            print(f'{count} tweets executed')
        # Make prediction. It consist of two attributes: probability [0-1] and sentiment [negative or positive].
        tweet = flair.data.Sentence(tweet)
        sentiment_model.predict(tweet)
        # Extract sentiment prediction
        # It is better to transform the values predicted into only one attribute with range [-1, 1] where -1 is
        # significantly negative and 1 is really positive
        if tweet.labels[0].value == 'NEGATIVE':
            results.append(-tweet.labels[0].score)
        else:
            results.append(tweet.labels[0].score)
    # Add a column with the sentiment predictions
    tweets['Sentiment'] = results
    # If likes, replies or retweets are supposed to strengthen the sentiment the results must be weighted accordingly
    if like_weight + reply_weight + retweet_weight != 0:
        tweets['Sentiment'] = tweets['Sentiment'] * (tweets['likes_count'] * like_weight +
                                                     tweets['replies_count'] * reply_weight +
                                                     tweets['retweets_count'] * retweet_weight)
    # Group by day returning the day average of the sentiment attribute
    tweets = tweets.groupby([tweets['date'].dt.date])['Sentiment'].mean()
    # Add the missing days and set their sentiment value to 0 (i.e. neutral).
    tweets = tweets.asfreq('D')
    tweets = tweets.fillna(0)
    return tweets


def vader_analysis(tweets, like_weight, reply_weight, retweet_weight, target_column='tweet'):
    """
    Performs natural language processing on the text of the tweets passed by means of Vader. Then it returns the tweets
    with the associated results of the sentiment analysis.

    :param tweets: dataframe that contains the tweets.
    :param like_weight: influence of each like on the related tweet.
    :param reply_weight: influence of each reply on the related tweet.
    :param retweet_weight: influence of each retweet on the related tweet.
    :param target_column: column of the dataframe that contains the text of the tweets. default_value='tweet'
    :return: the tweets grouped by day with the average sentiment score.
    """
    download('vader_lexicon')
    results = []
    sentiment_model = SentimentIntensityAnalyzer()
    print('Vader analysis started...')
    for count, tweet in enumerate(tweets[target_column].to_list(), 1):
        if count % 1000 == 0:
            print(f'{count} tweets executed')
        # Extract sentiment prediction
        result = sentiment_model.polarity_scores(tweet)['compound']
        results.append(result)
    # Add a column with the sentiment predictions
    tweets['Sentiment'] = results
    # If likes, replies or retweets are supposed to strengthen the sentiment the results must be weighted accordingly
    if like_weight + reply_weight + retweet_weight != 0:
        tweets['Sentiment'] = tweets['Sentiment'] * (tweets['likes_count'] * like_weight +
                                                     tweets['replies_count'] * reply_weight +
                                                     tweets['retweets_count'] * retweet_weight)
    # Group by day returning the day average of the sentiment attribute
    tweets = tweets.groupby([tweets['date'].dt.date])['Sentiment'].mean()
    # Add the missing days and set their sentiment value to 0 (i.e. neutral).
    tweets = tweets.asfreq('D')
    tweets = tweets.fillna(0)
    return tweets

