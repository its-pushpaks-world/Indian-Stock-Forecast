
import streamlit as st
import pandas as pd
from data_preprocessing import fetch_stock_data
from sentiment_analysis import get_news_sentiment
from forecasting_models import prophet_forecast, arima_forecast
from lstm_model import train_lstm

st.set_page_config(page_title="Indian Stock Forecast Pro", layout="wide")
st.title("📊 Indian Stock Forecast Pro")

symbol = st.text_input("Enter NSE Symbol (e.g., RELIANCE)")

if symbol:
    try:
        df = fetch_stock_data(symbol)
        st.write("### Latest Data")
        st.dataframe(df.tail())

        # Forecasts
        forecast_p = prophet_forecast(df)
        forecast_a = arima_forecast(df)
        forecast_l = train_lstm(df)

        st.write("### Forecasts")
        st.write(f"Prophet forecast (Next week): ₹{forecast_p['yhat'].iloc[-1]:.2f}")
        st.write(f"ARIMA forecast (Next week): ₹{forecast_a[-1]:.2f}")
        st.write(f"LSTM forecast (Next day): ₹{forecast_l:.2f}")

        # Sentiment
        st.write("### Sentiment Analysis")
        sentiments = get_news_sentiment(symbol)
        for item in sentiments:
            st.write(f"**{item['text']}**")
            st.write(f"Positive: {item['positive']:.2f}, Neutral: {item['neutral']:.2f}, Negative: {item['negative']:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter an NSE stock symbol to start.")
