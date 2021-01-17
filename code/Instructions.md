# Instructions

## Setup

1. Install all the other packages appointed in the [README.md](https://github.com/EdoardoGruppi/DAPS_assignment) file using conda or pip.
2. To install Twint could be necessary to run the code below to be sure that the last version is installed.
   ```
   pip install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
   ```
3. Install flair. Note that to run Flair with GPU it is required to have torch with cuda enabled.

   ```
   pip install flair
   ```

   Optional:

   ```python
   import torch
   import flair
   device = None
   if torch.cuda.is_available():
       device = torch.device('cuda:0')
   else:
       device = torch.device('cpu')
   print(device)
   torch.zeros(1).cuda()
   ```

4. When installing fbprophet it is needed to run the following lines since there is a conflict between specific version of fbprophet and pystan.
   ```
   conda install -c conda-forge fbprophet
   conda install -c conda-forge pystan
   ```
5. For the interaction with the mongo db cloud database it may be necessary to install the dnspython package.
   ```
   pip install dnspython
   ```

## Run the code

Once all the necessary packages have been installed you can run the code by typing this line on the terminal.

```
python main.py
```

**Note:** the main code execution is described phase by phase in the dedicated section below.

## Main execution

### Stock data acquisition and storage

Rather than running the data_gathere.py script it is preferable to downlad the datasets from the cloud database link provided. Indeed, the data acquistion process through the first solution requires at least 70-80 minutes (more than 10 times the time required by the second solution). Moreover, the code is mainly based on the datasets version provided by the MongoDB database link. Otherwise, there may be some little differences that can create conflicts.

```python
# Download datasets from Mongo DB
download_datasets()
# Once the datasets are locally available load the dataframes related to every pkl file.
indexes_dir = os.path.join(base_dir, 'Indexes.pkl')
time_series_dir = os.path.join(base_dir, 'Time_series.pkl')
covid_dir = os.path.join(base_dir, 'Covid.pkl')
tweets_dir = os.path.join(base_dir, 'MSFT_twitter.pkl')
news_dir = os.path.join(base_dir, 'News.pkl')
```

**Important:** after downloading the dataset the Datasets folder must be as follows:
![image](https://user-images.githubusercontent.com/48513387/104847643-98d65000-58e1-11eb-8806-d8d6383f8f49.png)

### Data preprocessing

In this phase the datasets before obtained are processed according to the content hosted. Whereas the tweets dataset is elaborated through the VADER tool, the news preprocessing involves a FLAIR pre-trained model.

```python
stock_data = time_series_preprocessing(time_series_dir, indexes_dir, path=True)
# Ohlc chart will be saved inside the code folder. See ohlc.png file.
ohlc_chart(path=time_series_dir, candle_size='10D', start=starting_date, end=ending_date, volume=False)
tweets = tweet_preprocessing(tweets_dir, analysis='vader', like_weight=0, reply_weight=0, retweet_weight=0, move=21)
news = tweet_preprocessing(news_dir, analysis='flair', like_weight=0, reply_weight=0, retweet_weight=0, move=21)
covid = covid_preprocessing(covid_dir, daily_change=True)
```

As last step of the pre-processing phase the datasets are integrated together via the apposite function. Then the unified dataset is divided in three parts for: training, validation and test.

```python
# Change the column names that can create conflicts
news.name = 'Mood'
# Create a unified dataset
dataframe = combine_dataset([stock_data, tweets, news, covid])
train, valid, test = dataset_division(dataframe)
del time_series_dir, covid_dir, news_dir, tweets_dir, indexes_dir
```

### Data exploration and Hypothesis testing

In the first part of this phase several functions are executed to visualize the composition of the dataset and the relationships between its variables.

```python
dataframe, new_dataframe, columns = change_format(concat([train, valid]))
# Search seasonality in the data
decompose_series(dataframe.Close, mode='multiplicative')
attributes_visualization(new_dataframe, columns, hue=['Day', 'Month', 'Year', 'Quarter', 'WeekDay'])
plot_auto_correlation(dataframe.Close, partial=False, lags=365)
# Visualize attribute relationships
attributes_visualization(new_dataframe, columns, hue=None)
multivariate_visualization(dataframe)
scatter_plot(dataframe, ['Mood', 'Close'])
check_stationarity(dataframe['Close'])
```

The second step of the exploration phase consists in testing some hypothesis that can be formulated throughout the entire process.

```python
dataframe_pct = percentage_change(dataframe, ['Covid'])
# Independence of the observations so that there is no relationship between the observations in each group.
plot_auto_correlation(dataframe_pct.Close)
print('\nHypothesis Testing...')
custom_test_1(dataframe_pct.Close, dataframe_pct.Sentiment, threshold=0.10, significance_level=0.06, test=1)
custom_test_2(dataframe_pct.Close, dataframe.Volume, percentile=65, test=2)
custom_test_2(dataframe_pct.Close, dataframe_pct.Volume, percentile=10, test=2)
custom_test_1(dataframe_pct.Sentiment, dataframe.Covid, threshold=0.35, significance_level=0.06, test=2)
custom_test_2(dataframe_pct.Close, dataframe_pct['S&p 100'], percentile=0.10, test=1)
custom_test_1(dataframe_pct.Volume, dataframe.Covid, threshold=0.90, significance_level=0.05, test=2)
custom_test_2(dataframe_pct.Volume, dataframe_pct.Sentiment, percentile=50, test=1)
custom_test_2(dataframe_pct.Sentiment, dataframe.Volume, percentile=40, significance_level=0.07, test=2)
custom_test_2(dataframe.Covid, dataframe_pct.Sentiment, percentile=90, test=0)
custom_test_2(dataframe_pct.Close, dataframe.Sentiment, percentile=50, test=1)
```

Then, the variables that turned out to be not really useful can be discarded. In the following lines the train, validation and test sets are also obtained to perform the forecasting starting from the sole company's stock data. Then the variables used are normalized, scaled or reduced according to which dataset is considered.

```python
del new_dataframe, dataframe, columns
train = train.drop(['Mood'], axis=1)
valid = valid.drop(['Mood'], axis=1)
test = test.drop(['Mood'], axis=1)
train_stock, valid_stock, test_stock = dataset_division(stock_data.drop(['S&p 100'], axis=1))
train_stock, valid_stock, test_stock = transform_dataset(train_stock, valid_stock, test_stock, reduction=False)
train, valid, test = transform_dataset(train, valid, test, algorithm='pca', n_components=0.90, reduction=True)
```

### Data inference

The following commented lines describe the validation phase in which the model hyperparameters are selected.

```python
# DATA INFERENCE =======================================================================================================
# Selection of the hyper-parameters
# Validate and compare models using only company's stock data
# prophet_predictions(train_stock, valid_stock, regressor=True, mode='multiplicative', holidays=False)
# arima = arima_predictions(train_stock, valid_stock, regressor=True)
# The validation split is used to set all the hyper-parameters that cannot be found with grid search algorithms
```

Since the ARIMA model hyper-parameters are obtained through cross-validation via the pmdarima function, the previous lines should be uncommented. Nevertheless, since the results of the function does not change in this case, the cross-validation is applied on the entire training dataset to reduce the total execution time and the amount of images displayed that can create a little of disorder.

```python
train_stock = concat([train_stock, valid_stock])
prophet_predictions(train_stock, test_stock, regressor=True, mode='multiplicative', holidays=False)
arima = arima_predictions(train_stock, test_stock, regressor=True)
# Remake the prediction using the model that led to best results in the previous step
train = concat([train, valid])
arima_test(model=arima, train=train, test=test, regressor=True)
```

## Additional information

A more detailed overview of the project is provided in the report published (the pdf file).

## Issues
