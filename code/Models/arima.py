# Import packages
from Modules.utilities import decompose_series, check_stationarity
from pmdarima.arima import auto_arima, ndiffs, nsdiffs
from pandas import DataFrame, concat
import seaborn as sn
import matplotlib.pyplot as plt
from Modules.utilities import metrics, residuals_properties
import numpy as np


# SARIMAX(p,d,q)x(P,D,Q)[lag]
# p and seasonal P: indicate the number of AR terms (lags of the stationary series)
# d and seasonal D: indicate differencing that must be done to stationarize series
# q and seasonal Q: indicate the number of MA terms (lags of the results errors)
# lag: indicates the seasonal length in the data
# To find the optimal parameters two approaches: grid-search algorithm or de-trending and differencing until the adf
# test allows to reject the null hypothesis.
def arima_predictions(train, test, regressor=True):
    """
    Performs forecasting using the ARIMA model selected after evaluating several configurations.

    :param train: training dataset.
    :param test: test dataset.
    :param regressor: if True the predictions are influenced also by the exogenous variables. default_value=True
    :return: the best model selected from the ARIMA's family
    """
    if regressor:
        # Select the columns related to the additional variables
        columns = [col for col in train.columns if col != 'Close']
        # The exogenous variables
        exog_train = train[columns]
        exog_test = test[columns]
    else:
        exog_train = None
        exog_test = None
    # Auto_arima function allows to set a range of p,d,q,P,D,and Q values and then fit models for all the possible
    # combinations. Then the model will keep the combination that reported back the best (lower) AIC value.
    # Arima differences both the y variable and the exogenous variables as specified in the arguments.
    # Find and fit the best model
    model = auto_arima(y=train['Close'], X=exog_train, start_p=1, start_q=1, max_d=12, seasonal=False, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    print(model.summary())
    # Specify the number of days in the future to predict
    periods = test.shape[0]
    # Make the prediction with the selected model
    results, conf = model.predict(X=exog_test, n_periods=periods, return_conf_int=True)
    # Visualize the results obtained
    arima_results(results, conf, train, test, model.resid())
    return model


def arima_test(model, train, test, regressor=True):
    """
    Performs forecasting using the ARIMA model provided.

    :param model: the ARIMA model selected previously.
    :param train: training dataset.
    :param test: test dataset.
    :param regressor: if True the predictions are influenced also by the exogenous variables. default_value=True
    :return:
    """
    if regressor:
        # Select the columns related to the additional variables
        columns = [col for col in train.columns if col != 'Close']
        # The exogenous variables
        exog_train = train[columns]
        exog_test = test[columns]
    else:
        exog_train = None
        exog_test = None
    # Fit the model provided
    model = model.fit(y=train['Close'], X=exog_train)
    # Specify the number of days in the future to predict
    periods = test.shape[0]
    # Forecast future values
    results, conf = model.predict(X=exog_test, n_periods=periods, return_conf_int=True)
    # Visualize the results obtained
    arima_results(results, conf, train, test, model.resid())


def arima_results(results, conf, data_train, data_test, residual):
    """
    Function to evaluate the results obtained by the ARIMA model through: the analysis of residuals, several metrics and
    multiple plots that represent the relationship between the true and the forecasted values.

    :param results: values predicted by the model.
    :param conf: confidence intervals of the values predicted.
    :param data_train: training dataset.
    :param data_test: test dataset.
    :param residual: residuals of the model.
    :return:
    """
    # Create a dataframe related to all the results with the true and forecasted values along with the confidence
    # intervals of the predictions
    results = results.reshape(-1, 1)
    results = np.concatenate((results, conf), axis=1)
    results = DataFrame(results, columns=['y_hat', 'y_hat_lower', 'y_hat_upper'])
    results.index = data_test.index
    results.loc[:, 'y'] = data_test.loc[:, 'Close']
    # Check that no value is below 0 even if in this case is not strictly necessary
    results.loc[:, 'y_hat'] = results.y_hat.clip(lower=0)
    results.loc[:, 'y_hat_lower'] = results.y_hat_lower.clip(lower=0)
    results.loc[:, 'y_hat_upper'] = results.y_hat_upper.clip(lower=0)
    # Plot comparison between forecasting results and predictions
    sn.set()
    f, ax = plt.subplots(figsize=(14, 8))
    ax.plot(data_train.index, data_train.Close, 'ko', markersize=3)
    ax.plot(results.index, results.y, 'ro', markersize=3)
    ax.plot(results.index, results.y_hat, color='coral', lw=1)
    ax.fill_between(results.index, results.y_hat_lower, results.y_hat_upper, color='coral', alpha=0.3)
    ax.axvline(results.head(1).index, color='k', ls='--', alpha=0.7)
    plt.show()
    # Plot comparison focusing on the period of the days predicted
    # The plot will display the last part of the series
    sn.set()
    f, ax = plt.subplots(figsize=(12, 6))
    samples = data_test.shape[0] * 5
    ax.plot(data_train.tail(samples).index, data_train.tail(samples).Close, marker='o', markersize=4, color='k')
    ax.plot(results.index, results.y, marker='o', markersize=4, color='r')
    ax.plot(results.index, results.y_hat, color='coral', lw=2)
    ax.fill_between(results.index, results.y_hat_lower, results.y_hat_upper, color='coral', alpha=0.3)
    ax.axvline(results.head(1).index, color='k', ls='--', alpha=0.7)
    plt.show()
    # Joint plot between the true and predicted values
    sn.jointplot(x='y_hat', y='y', data=results, kind="reg", color="b")
    plt.xlabel('Predictions')
    plt.ylabel('Observations')
    plt.show()
    # Visualize residuals properties
    residuals_properties(residual)
    # Evaluate model prediction capability through a series of metrics
    metrics(results.y, results.y_hat)



