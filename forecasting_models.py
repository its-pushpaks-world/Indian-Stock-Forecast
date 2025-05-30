
from prophet import Prophet

def prophet_forecast(df, periods=7):
    df_prophet = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    m = Prophet(daily_seasonality=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast
