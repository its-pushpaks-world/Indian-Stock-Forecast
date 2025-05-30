
from prophet import Prophet
import pandas as pd
import numpy as np

def prophet_forecast(df, periods=7):
    df_prophet = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet(daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods+14, freq='B')  # Extra buffer
    forecast = m.predict(future)
    forecast = forecast[forecast['ds'].dt.weekday < 5]  # Remove weekends
    indian_holidays = pd.to_datetime(["2025-01-26", "2025-08-15", "2025-10-02", "2025-11-12", "2025-12-25"])
    forecast = forecast[~forecast['ds'].isin(indian_holidays)]
    forecast = forecast.tail(periods)  # Get final forecast period
    return forecast

def evaluate_model(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return mape, rmse
