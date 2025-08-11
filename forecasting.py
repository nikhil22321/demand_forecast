
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def prophet_forecast(df, periods=30):
    # df must have columns ds (date) and y (value)
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def arima_forecast(series, periods=30, order=(5,1,0)):
    # series: pandas Series indexed by date
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    fc = model_fit.get_forecast(steps=periods)
    mean = fc.predicted_mean
    conf = fc.conf_int()
    df = pd.DataFrame({
        'ds': pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D'),
        'yhat': mean.values,
        'yhat_lower': conf.iloc[:,0].values,
        'yhat_upper': conf.iloc[:,1].values
    })
    return df
