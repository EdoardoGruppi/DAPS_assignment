# Import Packages
import flair
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download


def flair_analysis(tweets, like_weight, reply_weight, retweet_weight, target_column='tweet'):
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
    tweets['sentiment'] = results
    # If likes, replies or retweets are supposed to strengthen the sentiment the results must be weighted accordingly
    if like_weight + reply_weight + retweet_weight != 0:
        tweets['sentiment'] = tweets['sentiment'] * (tweets['likes_count'] * like_weight +
                                                     tweets['replies_count'] * reply_weight +
                                                     tweets['retweets_count'] * retweet_weight)
    # Group by day returning the day average of the sentiment attribute
    tweets = tweets.groupby([tweets['date'].dt.date])['sentiment'].mean()
    return tweets


def vader_analysis(tweets, like_weight, reply_weight, retweet_weight, target_column='tweet'):
    download('vader_lexicon')
    results = []
    sentiment_model = SentimentIntensityAnalyzer()
    print('Vader analysis started...')
    for count, tweet in enumerate(tweets[target_column].to_list(), 1):
        if count % 500 == 0:
            print(f'{count} tweets executed')
        # Extract sentiment prediction
        result = sentiment_model.polarity_scores(tweet)['compound']
        results.append(result)
    # Add a column with the sentiment predictions
    tweets['sentiment'] = results
    # If likes, replies or retweets are supposed to strengthen the sentiment the results must be weighted accordingly
    if like_weight + reply_weight + retweet_weight != 0:
        tweets['sentiment'] = tweets['sentiment'] * (tweets['likes_count'] * like_weight +
                                                     tweets['replies_count'] * reply_weight +
                                                     tweets['retweets_count'] * retweet_weight)
    # Group by day returning the day average of the sentiment attribute
    tweets = tweets.groupby([tweets['date'].dt.date])['sentiment'].mean()
    return tweets

