
from textblob import TextBlob
import yfinance as yf

def get_news_sentiment(ticker):
    news = yf.Ticker(ticker).news[:5]
    results = []
    for item in news:
        txt = item.get("title","") + " " + item.get("summary","")
        sentiment = TextBlob(txt).sentiment.polarity
        if sentiment > 0.05:
            label = "Positive"
        elif sentiment < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        results.append({"text": txt, "sentiment": label, "score": sentiment})
    return results
