
import streamlit as st
from data_preprocessing import fetch_stock_data
from sentiment_analysis import get_news_sentiment
from forecasting_models import prophet_forecast, evaluate_model
import numpy as np
import pandas as pd

st.set_page_config(page_title="Indian Stock Forecast Pro", layout="centered")
st.title("📊 Indian Stock Forecast Pro")

ticker = st.text_input("Enter NSE symbol (e.g., RELIANCE)").upper().strip()
if ticker:
    try:
        df = fetch_stock_data(ticker)
        last_close = df['Close'].iloc[-1]
        min_close = df['Close'].min() * 0.9

        forecast = prophet_forecast(df, periods=7)
        forecast = forecast[forecast['ds'].dt.weekday < 5]  # Remove weekends

        # Remove public holidays (sample list, can be extended)
        indian_holidays = pd.to_datetime([
            "2025-01-26", "2025-08-15", "2025-10-02", "2025-11-12", "2025-12-25"
        ])
        forecast = forecast[~forecast['ds'].isin(indian_holidays)]

        # Clip to 90% of minimum close to avoid unrealistic negative dips
        forecast['yhat'] = forecast['yhat'].clip(lower=min_close)

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
        if not sentiments:
            st.markdown("**News:** No news available")
        else:
            scores = [item.get("score", 0) for item in sentiments]
            all_similar = all(abs(scores[0] - s) < 0.01 for s in scores)

            if all_similar:
                item = sentiments[0]
                score = item.get("score", 0)
                st.markdown(f"**News:** {item.get('text', 'N/A')}")
                st.markdown(f"**Sentiment:** {item.get('sentiment', '')}")
                st.markdown(f"**Score:** {score:.2f}")
            else:
                for item in sentiments:
                    score = item.get("score", 0)
                    st.markdown(f"**News:** {item.get('text', 'N/A')}")
                    st.markdown(f"**Sentiment:** {item.get('sentiment', '')}")
                    st.markdown(f"**Score:** {score:.2f}")
                    st.write("---")

        st.subheader("Forecast Chart")
        st.line_chart(forecast.set_index('ds')['yhat'])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter a ticker to start.")
