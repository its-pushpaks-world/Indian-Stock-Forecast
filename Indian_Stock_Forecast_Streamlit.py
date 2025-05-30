
import streamlit as st
from data_preprocessing import fetch_stock_data
from sentiment_analysis import get_news_sentiment
from forecasting_models import prophet_forecast, evaluate_model
import numpy as np
import datetime

st.set_page_config(page_title="Indian Stock Forecast Pro", layout="centered")
st.title("📊 Indian Stock Forecast Pro")

ticker = st.text_input("Enter NSE symbol (e.g., RELIANCE)").upper().strip()
if ticker:
    try:
        df = fetch_stock_data(ticker)
        forecast = prophet_forecast(df, periods=7)
        last_close = df['Close'].iloc[-1]
        forecast_today = forecast['yhat'].iloc[-6]
        forecast_tomorrow = forecast['yhat'].iloc[-5]
        forecast_next_week = forecast['yhat'].iloc[-1]

        recommendation = "BUY" if forecast_next_week > last_close * 1.02 else "SELL" if forecast_next_week < last_close * 0.98 else "HOLD"

        st.markdown(f"### Recommendation: **{recommendation}**")
        st.markdown(f"**Current Price:** ₹{last_close:.2f}")
        st.markdown(f"**Forecast Today:** ₹{forecast_today:.2f}")  
        st.markdown(f"**Forecast Tomorrow:** ₹{forecast_tomorrow:.2f}")  
        st.markdown(f"**Forecast Next Week:** ₹{forecast_next_week:.2f}")

        st.subheader("Sentiment Analysis")
        for item in get_news_sentiment(ticker+".NS"):
            st.write(item["text"])
            score = item.get("score",0)
            st.write(f'{item.get("sentiment","")} (Score: {score:.2f})')
            st.write("---")

        st.subheader("Forecast Chart")
        st.line_chart(forecast.set_index('ds')['yhat'])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter a ticker to start.")
