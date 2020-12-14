# Import packages
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import to_datetime, concat, DataFrame
from Modules.config import *
from Modules.utilities import metrics, residuals


# fbprophet implements models a time-series as the sum (or multiplication) of different components (trends,
# periodic components, holidays and special events) allowing to incorporate additional regressors taken from outer
# sources. The main reference is Taylor and Letham, 2017
def prophet_predictions(train, test, validation=True, periods=31, regressor=True):
    data = prepare_data(train, 'Close')
    additional_info = DataFrame()
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=True, interval_width=0.95)
    if regressor:
        for column in train:
            if column != 'Close':
                model.add_regressor(column)
                additional_info[column] = concat([train[column], test[column]])
    model.fit(data)
    # We need to specify the number of days in future
    if validation:
        periods = test.shape[0]
    future = model.make_future_dataframe(periods=periods, freq='1D')
    if regressor:
        additional_info = additional_info.reset_index().rename(columns={'date': 'ds'}).drop('ds', axis=1)
        future = concat([future, additional_info], axis=1)
    forecast = model.predict(future)
    sn.set()
    model.plot_components(forecast)
    plt.show()
    results = prophet_results(forecast, train, test)


def prepare_data(data, target_feature):
    new_data = data.copy()
    new_data.reset_index(inplace=True)
    new_data = new_data.rename({'date': 'ds', f'{target_feature}': 'y'}, axis=1)
    return new_data


def prophet_results(forecast, data_train, data_test):
    """
    Function to convert the output Prophet dataframe to a datetime index and append the actual target values at the end
    """
    forecast.index = to_datetime(forecast.ds)
    data = concat([data_train, data_test], axis=0)
    forecast.loc[:, 'y'] = data.loc[:, 'Close']
    forecast.loc[:, 'yhat'] = forecast.yhat.clip(lower=0)
    forecast.loc[:, 'yhat_lower'] = forecast.yhat_lower.clip(lower=0)
    forecast.loc[:, 'yhat_upper'] = forecast.yhat_upper.clip(lower=0)

    # Plot comparison between forecasting results and predictions
    sn.set()
    f, ax = plt.subplots(figsize=(14, 8))
    end_train = data_train.tail(1).index[0]
    train = forecast.loc[starting_date:end_train, :]
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=1)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    start_test = data_test.head(1).index[0]
    test = forecast.loc[start_test:, :]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=1)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc[start_test, 'ds'], color='k', ls='--', alpha=0.7)
    plt.show()

    # Joint plot
    sn.jointplot(x='yhat', y='y', data=train, kind="reg", color="b")
    plt.xlabel('Predictions')
    plt.ylabel('Observations')
    plt.show()
    sn.jointplot(x='yhat', y='y', data=test, kind="reg", color="b")
    plt.xlabel('Predictions')
    plt.ylabel('Observations')
    plt.show()

    metrics(test.y, test.yhat)
    residuals(test.y, test.yhat)
    return forecast
