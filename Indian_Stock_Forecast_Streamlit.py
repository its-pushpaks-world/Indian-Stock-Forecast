
import streamlit as st
from data_preprocessing import fetch_stock_data
from sentiment_analysis import get_news_sentiment
from forecasting_models import prophet_forecast, evaluate_model
import numpy as np

st.set_page_config(page_title="Indian Stock Forecast Pro", layout="centered")
st.title("📊 Indian Stock Forecast Pro")

ticker = st.text_input("Enter NSE symbol (e.g., RELIANCE)").upper().strip()
if ticker:
    try:
        df = fetch_stock_data(ticker)
        last_close = df['Close'].iloc[-1]
        min_price = last_close * 0.1  # Forecast floor at 10% of last close
        forecast = prophet_forecast(df, periods=7)
        forecast['yhat'] = forecast['yhat'].clip(lower=min_price)

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
        sentiments = get_news_sentiment(ticker+".NS")
        scores = [item.get("score", 0) for item in sentiments]
        all_similar = all(abs(scores[0] - s) < 0.01 for s in scores)

        if all_similar:
            st.write(sentiments[0]["text"])
            st.write(f'{sentiments[0].get("sentiment","")} (Score: {scores[0]:.2f})')
        else:
            for item in sentiments:
                score = item.get("score", 0)
                st.write(item["text"])
                st.write(f'{item.get("sentiment","")} (Score: {score:.2f})')
                st.write("---")

        st.subheader("Forecast Chart")
        st.line_chart(forecast.set_index('ds')['yhat'])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter a ticker to start.")
