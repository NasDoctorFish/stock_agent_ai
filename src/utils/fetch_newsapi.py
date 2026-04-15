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
    import socket
    from transformers import pipeline

    # DNS pre-check — avoids per-ticker spam if the host is unreachable
    try:
        socket.getaddrinfo("newsapi.org", 443, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror as dns_err:
        print(f"[NewsAPI] DNS resolution failed — skipping sentiment fetch. ({dns_err})")
        return pd.DataFrame()

    print(f"[Process {os.getpid()}] Loading FinBERT Model...")
    sentiment = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

    records = []
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    for t in tqdm(tickers, desc="Fetching news + sentiment"):
        q = f"{t} stock"
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={q}&from={start_date}&to={end_date}&language=en&apiKey={NEWS_API_KEY}"
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            consecutive_failures = 0
            for art in r.json().get("articles", []):
                sent = sentiment(art["title"])
                records.append({
                    "stock_id": t,
                    "date": art["publishedAt"][:10],
                    "source": art["source"]["name"],
                    "headline": art["title"],
                    "sentiment_label": sent[0]["label"],
                    "sentiment_score": sent[0]["score"],
                })
        except requests.exceptions.ConnectionError as e:
            consecutive_failures += 1
            print(f"[NewsAPI] Connection error for {t}: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("[NewsAPI] Too many connection errors — aborting.")
                break
        except Exception as e:
            print(f"[NewsAPI] Error fetching {t}: {e}")

    return pd.DataFrame(records)

# Rename this function to clarify it only gets text
def fetch_news_text_only(tickers, start_date, end_date, NEWS_API_KEY):
    # REMOVE: from transformers import pipeline
    # REMOVE: sentiment = pipeline(...)

    import socket

    # Quick DNS probe before looping over all tickers.
    # If newsapi.org is unreachable, bail out immediately instead of
    # printing the same connection error for every ticker.
    try:
        socket.getaddrinfo("newsapi.org", 443, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror as dns_err:
        print(f"[NewsAPI] DNS resolution failed for newsapi.org — skipping news fetch. ({dns_err})")
        return pd.DataFrame()

    records = []
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3  # abort after 3 back-to-back network errors

    for t in tqdm(tickers, desc="Fetching news data"):
        q = f"{t} stock"
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={q}&from={start_date}&to={end_date}&language=en&apiKey={NEWS_API_KEY}"
        )

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            consecutive_failures = 0  # reset on success

            for art in data.get("articles", []):
                records.append({
                    "stock_id": t,
                    "date": art["publishedAt"][:10],
                    "source": art["source"]["name"],
                    "headline": art["title"],
                    "sentiment_label": None,
                    "sentiment_score": 0.0,
                })

        except requests.exceptions.ConnectionError as e:
            consecutive_failures += 1
            print(f"[NewsAPI] Connection error for {t}: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[NewsAPI] {MAX_CONSECUTIVE_FAILURES} consecutive connection failures — "
                    "aborting news fetch to avoid spamming errors."
                )
                break

        except requests.exceptions.Timeout:
            consecutive_failures += 1
            print(f"[NewsAPI] Timeout for {t}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("[NewsAPI] Too many timeouts — aborting news fetch.")
                break

        except Exception as e:
            # Non-network errors (bad JSON, HTTP 4xx/5xx) — log and continue
            print(f"[NewsAPI] Error fetching {t}: {e}")

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
   