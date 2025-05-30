
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

def fetch_stock_data(ticker, period="1y", interval="1d"):
    if not ticker.endswith(".NS"):
        ticker += ".NS"
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    df.index = df.index.tz_convert(None)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['EMA200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Hist'] = macd.macd_diff()
    df.dropna(inplace=True)
    return df
