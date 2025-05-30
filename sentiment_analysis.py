
from textblob import TextBlob
import yfinance as yf

def get_news_sentiment(ticker):
    news = yf.Ticker(ticker).news[:5]
    results = []
    for item in news:
        txt = item.get("title","")
        sentiment = TextBlob(txt).sentiment.polarity
        label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        results.append({"text": txt, "sentiment": label, "score": sentiment})
    return results
