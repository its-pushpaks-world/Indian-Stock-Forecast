
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
        st.subheader("Latest Data")
        st.dataframe(df.tail())

        st.subheader("Sentiment Analysis")
        for item in get_news_sentiment(ticker+".NS"):
            st.write(item["text"])
            st.write(f'{item["sentiment"]} (Score: {item["score"]:.2f})')
            st.write("---")

        forecast = prophet_forecast(df)
        pred_today = forecast['yhat'].iloc[-(7+1)]
        pred_next = forecast['yhat'].iloc[-1]
        actuals = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
        merged = actuals.set_index('ds').join(forecast.set_index('ds')[['yhat']]).dropna()
        eval_df = merged.tail(30)
        mape, rmse = evaluate_model(eval_df['y'], eval_df['yhat'])

        st.subheader("Forecasts")
        st.write(f"Current Price: ₹{df['Close'].iloc[-1]:.2f}")
        st.write(f"Forecast Today: ₹{pred_today:.2f}")
        st.write(f"Forecast Next Week: ₹{pred_next:.2f}")
        st.subheader("Accuracy (Last 30 days)")
        st.write(f"MAPE: {mape:.2f}%")
        st.write(f"RMSE: {rmse:.2f}")

        st.subheader("Forecast Chart")
        st.line_chart(forecast.set_index('ds')['yhat'])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter a ticker to start.")
