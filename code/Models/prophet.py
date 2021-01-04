# Import packages
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import to_datetime, concat, DataFrame, merge
from Modules.config import *
from Modules.utilities import metrics, decompose_series, residuals_properties


def prophet_predictions(train, test, regressor=True, mode='multiplicative', holidays=False, sps=10, cps=0.1,
                        interval=0.95, n_change_points=25):
    """
    Facebook Prophet models a time-series as the sum (or multiplication) of different components (trends,
    periodic components, holidays and special events) allowing to incorporate additional regressors taken from outer
    sources. The models parameters are explained in: https://bit.ly/3p6Ek5n.

    :param train: training dataset.
    :param test: test dataset.
    :param regressor: if True the predictions are influenced also by the exogenous variables. default_value=True
    :param mode: seasonality mode. It can be additive or multiplicative. default_value='multiplicative'
    :param holidays: if True, Prophet uses also US holidays to predict values. default_value=False
    :param sps: seasonality_prior_scale. Parameter to modulate the strength of the seasonality model. default_value=10
    :param cps: parameter to modulate the flexibility of the automatic change point selection. default_value=0.1
    :param interval: width of the uncertainty intervals provided for the forecast. default_value=0.95
    :param n_change_points: number of potential change points to include. default_value=25
    :return:
    """
    exogenous = None
    # Prepare data as required by Prophet
    data = prepare_data(train, 'Close')
    model = Prophet(seasonality_mode=mode, yearly_seasonality=False, weekly_seasonality=False,
                    interval_width=interval, daily_seasonality=False, seasonality_prior_scale=sps,
                    changepoint_prior_scale=cps, n_changepoints=n_change_points)
    # Add seasonality to the model
    # model.add_seasonality(name='two-years', period=700, fourier_order=20, prior_scale=15)
    # If requested add holidays component to the model
    if holidays:
        model.add_country_holidays(country_name='US')
    # If requested add exogenous variables to the model
    if regressor:
        # Select the columns related only to the exogenous variables
        columns = [col for col in train.columns if col != 'Close']
        exogenous = concat([train[columns], test[columns]])
        # Add each additional variable as a regressor
        for col in columns:
            model.add_regressor(col)
    # Train the model
    model.fit(data)
    # Specify the number of days in the future to predict
    future = model.make_future_dataframe(periods=test.shape[0], freq='1D')
    if regressor:
        # Add the regressor values within the test period
        exogenous = exogenous.reset_index().rename(columns={'date': 'ds'}).drop('ds', axis=1)
        future = concat([future, exogenous], axis=1)
    # Make the prediction with the Prophet model
    forecast = model.predict(future)
    # Plot the components
    sn.set()
    model.plot_components(forecast)
    plt.show()
    # Visualize the results obtained
    prophet_results(forecast, train, test)


def prepare_data(data, target_feature):
    """
    Facebook Prophet requires the inputs to be in a particular format. The time-series to forecast must be called 'y'
    and the date column 'ds'

    :param data: training dataset.
    :param target_feature: name of the column associated to the time series.
    :return: the input data transformed.
    """
    new_data = data.copy()
    new_data.reset_index(inplace=True)
    # Rename column to obtain the format required by Prophet
    new_data = new_data.rename({'date': 'ds', f'{target_feature}': 'y'}, axis=1)
    return new_data


def prophet_results(forecast, data_train, data_test):
    """
    Function to evaluate the results obtained by the ARIMA model through: the analysis of residuals, several metrics and
    multiple plots that represent the relationship between the true and the forecasted values.

    :param forecast: dataframe obtained by calling the predict function with the Prophet model.
    :param data_train: training dataset.
    :param data_test: test dataset.
    :return:
    """
    forecast.index = to_datetime(forecast.ds)
    data = concat([data_train, data_test], axis=0)
    # Add the Close values to the forecast dataframe
    forecast.loc[:, 'y'] = data.loc[:, 'Close']
    # Check that no value is below 0 even if in this case is not strictly necessary
    forecast.loc[:, 'yhat'] = forecast.yhat.clip(lower=0)
    forecast.loc[:, 'yhat_lower'] = forecast.yhat_lower.clip(lower=0)
    forecast.loc[:, 'yhat_upper'] = forecast.yhat_upper.clip(lower=0)
    # Plot comparison between forecasting results and predictions
    sn.set()
    f, ax = plt.subplots(figsize=(14, 8))
    end_train = data_train.tail(1).index[0]
    train = forecast.loc[starting_date:end_train, :]
    residuals = train.y - train.yhat
    ax.plot(train.index, train.y, 'ko', markersize=3)
    ax.plot(train.index, train.yhat, color='steelblue', lw=1)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    start_test = data_test.head(1).index[0]
    test = forecast.loc[start_test:, :]
    ax.plot(test.index, test.y, 'ro', markersize=3)
    ax.plot(test.index, test.yhat, color='coral', lw=1)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc[start_test, 'ds'], color='k', ls='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Plot comparison focusing on the days predicted
    # The plot will display the last part of the series
    sn.set()
    f, ax = plt.subplots(figsize=(14, 8))
    samples = data_test.shape[0] * 5
    end_train = data_train.tail(1).index[0]
    train = forecast.loc[starting_date:end_train, :].tail(samples)
    ax.plot(train.index, train.y, marker='o', markersize=4, color='k')
    ax.plot(train.index, train.yhat, color='steelblue', lw=2)
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    start_test = data_test.head(1).index[0]
    test = forecast.loc[start_test:, :]
    ax.plot(test.index, test.y, marker='o', markersize=4, color='r')
    ax.plot(test.index, test.yhat, color='coral', lw=2)
    ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)
    ax.axvline(forecast.loc[start_test, 'ds'], color='k', ls='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # todo -- Joint plot between the true and predicted values
    # sn.jointplot(x='yhat', y='y', data=train, kind="reg", color="b")
    # plt.xlabel('Predictions')
    # plt.ylabel('Observations')
    # plt.show()
    # sn.jointplot(x='yhat', y='y', data=test, kind="reg", color="b")
    # plt.xlabel('Predictions')
    # plt.ylabel('Observations')
    # plt.tight_layout()
    # plt.show()
    # Visualize residuals properties
    residuals_properties(residuals)
    # Evaluate model prediction capability through a series of metrics
    metrics(test.y, test.yhat)
