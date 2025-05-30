
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import yfinance as yf

def get_news_sentiment(ticker):
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    news = yf.Ticker(ticker).news[:5]
    results = []
    for item in news:
        txt = item.get("title","") + " " + item.get("summary","")
        inputs = tokenizer(txt, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = softmax(outputs.logits.detach().numpy()[0])
        sentiment = {"text": txt, "positive": float(probs[0]), "neutral": float(probs[1]), "negative": float(probs[2])}
        results.append(sentiment)
    return results
