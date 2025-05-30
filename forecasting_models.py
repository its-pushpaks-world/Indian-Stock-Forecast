
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def prophet_forecast(df, periods=7):
    df_prophet = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet(daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast

def evaluate_model(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, rmse
