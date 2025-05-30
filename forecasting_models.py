
from prophet import Prophet
import pandas as pd
import numpy as np

def prophet_forecast(df, periods=7):
    df_prophet = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet(daily_seasonality=False)
    m.add_country_holidays(country_name='IN')
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods, freq='B')  # Business days only
    forecast = m.predict(future)
    return forecast

def evaluate_model(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mape, rmse
