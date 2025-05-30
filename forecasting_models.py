
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

def prophet_forecast(df):
    df_prophet = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return forecast

def evaluate_model(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return mape, rmse
