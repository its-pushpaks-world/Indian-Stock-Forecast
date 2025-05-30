
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
import datetime

st.set_page_config(page_title="Indian Stock Forecast", layout="centered")

st.title("📈 Indian Stock Forecasting Tool")

# User input
symbol = st.text_input("Enter NSE stock symbol (e.g., RELIANCE):").upper().strip()

if symbol:
    try:
        # Fetch data
        if not symbol.endswith(".NS"):
            symbol += ".NS"
        df = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=True)
        df.index = df.index.tz_convert(None)

        # Technical indicators
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Hist'] = macd.macd_diff()
        df.dropna(inplace=True)

        # Forecasts
        y = df['Close'].values
        X = np.arange(len(y))
        slope, intercept = np.polyfit(X, y, 1)
        forecast_today = slope * (len(y) - 1) + intercept
        forecast_next_week = slope * (len(y) + 6) + intercept

        weights = np.linspace(0.5, 1.5, len(y))
        exp_forecast = np.average(y, weights=weights)
        forecast_today = np.mean([forecast_today, exp_forecast])
        forecast_next_week = np.mean([forecast_next_week, exp_forecast])

        current_price = df['Close'].iloc[-1]

        # Trend
        last_week_return = (df['Close'].iloc[-1] - df['Close'].iloc[-7]) / df['Close'].iloc[-7] * 100 if len(df) >= 7 else 0
        if last_week_return < -2:
            trend = "DOWN"
        elif last_week_return > 2:
            trend = "UP"
        else:
            trend = "UP" if slope > 0 else "DOWN"

        st.subheader(f"Forecast for {symbol[:-3]}")
        st.write(f"**Current Price**: ₹{current_price:.2f}")
        st.write(f"**Forecast Today**: ₹{forecast_today:.2f}")
        st.write(f"**Forecast Next Week**: ₹{forecast_next_week:.2f}")
        st.write(f"**Trend Prediction**: {trend}")

        # Technical indicators
        st.write("**Technical Indicators**:")
        st.write(f"RSI: {df['RSI'].iloc[-1]:.2f}")
        st.write(f"EMA50: {df['EMA50'].iloc[-1]:.2f}")
        st.write(f"EMA200: {df['EMA200'].iloc[-1]:.2f}")
        st.write(f"MACD Histogram: {df['MACD_Hist'].iloc[-1]:.4f}")

        # Data preview
        if st.checkbox("Show recent data"):
            st.dataframe(df.tail())

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter an NSE stock symbol to start.")
