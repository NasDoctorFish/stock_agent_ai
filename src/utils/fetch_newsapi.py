import requests
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm


def fetch_news_sentiment(tickers, start_date, end_date, NEWS_API_KEY):
    from transformers import pipeline
    sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    records = []
    for t in tqdm(tickers, desc="Fetching news data"):
        q = f"{t} stock"
        url = f"https://newsapi.org/v2/everything?q={q}&from={start_date}&to={end_date}&language=en&apiKey={NEWS_API_KEY}"
        r = requests.get(url)
        for art in r.json().get("articles", []):
            # sent = TextBlob(art["title"]).sentiment.polarity # v.1 sentiment score
            # FinBERT: 증권/금융 뉴스용 감성 지수
            # 출력 예시
            # [{'label': 'negative', 'score': 0.994}]
            sent = sentiment(art["title"])
            records.append({
                "stock_id": t,
                "date": art["publishedAt"][:10],
                "source": art["source"]["name"],
                "headline": art["title"],
                "sentiment_label": sent[0]['label'],
                "sentiment_score": sent[0]['score']
            })
    return pd.DataFrame(records)


# # For testing
# from transformers import pipeline
# sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
# print(sentiment("The market looks terrible today."))
# print(sentiment("New K-drama reached 8 million views on Netflix"))
