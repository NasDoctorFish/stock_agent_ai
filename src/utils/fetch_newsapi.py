import os

# 1. Force single-threaded operation for the underlying C++ libraries
# (Put this at the VERY TOP of your main script, before any other imports)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import requests
import pandas as pd
from tqdm import tqdm
import multiprocessing


def fetch_news_sentiment(tickers, start_date, end_date, NEWS_API_KEY):
    # Import INSIDE the function to ensure it loads in the child process memory space
    from transformers import pipeline 
    
    print(f"[Process {os.getpid()}] Loading FinBERT Model...")
    # This load happens safely inside the independent child process
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

# Rename this function to clarify it only gets text
def fetch_news_text_only(tickers, start_date, end_date, NEWS_API_KEY):
    # REMOVE: from transformers import pipeline
    # REMOVE: sentiment = pipeline(...) 
    
    records = []
    for t in tqdm(tickers, desc="Fetching news data"):
        q = f"{t} stock"
        url = f"https://newsapi.org/v2/everything?q={q}&from={start_date}&to={end_date}&language=en&apiKey={NEWS_API_KEY}"
        
        try:
            r = requests.get(url, timeout=10) # Good practice to add timeout
            data = r.json()
            
            for art in data.get("articles", []):
                # Just save the raw text. No analysis yet.
                records.append({
                    "stock_id": t,
                    "date": art["publishedAt"][:10],
                    "source": art["source"]["name"],
                    "headline": art["title"],
                    # placeholders for now
                    "sentiment_label": None, 
                    "sentiment_score": 0.0
                })
        except Exception as e:
            print(f"Error fetching {t}: {e}")
            
    return pd.DataFrame(records)


def fetch_news_final(tickers, start_date, end_date, NEWS_API_KEY):
     # 1. FETCH DATA (Safe to use multiprocessing here if you are)
    print("Step 1: Fetching News Text...")
    
    # Assuming you run the fetcher here (either single process or multi)
    news_df = fetch_news_text_only(tickers, start_date, end_date, NEWS_API_KEY)
    
    print(f"✅ Fetched {len(news_df)} news headlines.")

    # 2. ANALYZE DATA (Run this sequentially in the main process)
    if not news_df.empty:
        print("Step 2: Loading AI Model (Main Process)...")
        from transformers import pipeline
        
        # Load model ONCE here
        sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", device=-1) # device=-1 for CPU, 0 for GPU
        
        print("Step 3: Calculating Sentiment...")
        
        # Batch processing is faster than a loop!
        headlines = news_df["headline"].tolist()
        
        # Process in batches to avoid memory issues if list is huge
        results = sentiment_pipeline(headlines, batch_size=16)
        
        # Assign results back to DataFrame
        news_df["sentiment_label"] = [r['label'] for r in results]
        news_df["sentiment_score"] = [r['score'] for r in results]
        
        print("✅ Sentiment Analysis Complete.")
    
    # 3. SAVE TO DB
    # Now save 'news_df' to your database...
    # db.save_to_duckdb(news_df, table_name="news_sentiment") already done outside
   