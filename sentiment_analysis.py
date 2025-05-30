
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import yfinance as yf

def get_news_sentiment(ticker):
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    news = yf.Ticker(ticker).news[:5]
    sentiments = []
    for item in news:
        text = item['title']
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits.detach().numpy()[0])
        sentiment = {"text": text, "positive": probs[0], "neutral": probs[1], "negative": probs[2]}
        sentiments.append(sentiment)
    return sentiments
