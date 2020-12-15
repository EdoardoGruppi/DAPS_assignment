# Import packages
from Modules.utilities import decompose_series, check_stationarity
from pmdarima.arima import auto_arima
from pandas import DataFrame, concat
import seaborn as sn
import matplotlib.pyplot as plt
from Modules.utilities import metrics, residuals
import numpy as np


# SARIMA(p,d,q)x(P,D,Q,lag)
# p and seasonal P: indicate the number of AR terms (lags of the stationary series)
# d and seasonal D: indicate differencing that must be done to stationarize series
# q and seasonal Q: indicate the number of MA terms (lags of the results errors)
# lag: indicates the seasonal length in the data
# To find the optimal parameters two approaches: grid-search algorithm or de-trending and differencing until the adf
# test allows to reject the null hypothesis.
def arima_predictions(train, test, regressor=True, mode='multiplicative', exog_train=None, exog_test=None):
    decompose_series(train['Close'], mode=mode)
    check_stationarity(train['Close'])
    if regressor:
        columns = [col for col in train.columns if col != 'Close']
        exog_train = train[columns]
        exog_test = test[columns]
    # auto_arima function allows to set a range of p,d,q,P,D,and Q values and then fit models for all the possible
    # combinations. Then the model will keep the combination that reported back the best AIC value.
    # m = 7 means daily seasonality, 12 monthly seasonality, 52 weekly seasonality
    model = auto_arima(y=train['Close'], X=exog_train, start_p=1, start_q=1, m=12, start_P=0, seasonal=True, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    print(model.summary())
    # We need to specify the number of days in future
    periods = test.shape[0]
    results, conf = model.predict(X=exog_test, n_periods=periods, return_conf_int=True)
    model.plot_diagnostics(lags=7, figsize=(8, 8))
    plt.show()
    arima_results(results, conf, train, test)


def arima_results(results, conf, data_train, data_test):
    results = results.reshape(-1, 1)
    results = np.concatenate((results, conf), axis=1)
    results = DataFrame(results, columns=['y_hat', 'y_hat_lower', 'y_hat_upper'])
    results.index = data_test.index
    results.loc[:, 'y'] = data_test.loc[:, 'Close']
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

    # Joint plot
    sn.jointplot(x='y_hat', y='y', data=results, kind="reg", color="b")
    plt.xlabel('Predictions')
    plt.ylabel('Observations')
    plt.show()

    metrics(results.y, results.y_hat)
    residuals(results.y, results.y_hat)
