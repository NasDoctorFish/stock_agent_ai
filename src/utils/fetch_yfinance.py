import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

def fetch_prices(tickers, start_date, end_date, delay=1.0, max_retries=3):
    """Download price + quarterly fundamentals safely with rate-limit awareness."""
    all_prices, all_fund = [], []
    
    for t in tqdm(tickers, desc="Fetching data"):
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(t)
                
                # ✅ Price Data
                df_price = stock.history(start=start_date, end=end_date)
                if df_price.empty:
                    print(f"⚠️ No price data for {t}")
                    break
                
                df_price = df_price.reset_index()
                df_price["stock_id"] = t
                df_price = df_price.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"
                })
                all_prices.append(df_price)

                # ✅ Fundamentals (quarterly)
                fin = stock.quarterly_financials
                if fin is None or fin.empty:
                    print(f"⚠️ No fundamentals for {t}")
                    break

                fin = fin.T.reset_index()
                fin["stock_id"] = t
                fin = fin.rename(columns={"index": "fiscal_date"})
                all_fund.append(fin)

                # 💨 break retry loop if success
                break

            except Exception as e:
                print(f"❌ Error fetching {t} (attempt {attempt+1}): {e}")
                time.sleep(delay * 2)
        time.sleep(delay)  # rate-limit control
    
    # ✅ Merge collected data
    price_df = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
    fund_df = pd.concat(all_fund, ignore_index=True) if all_fund else pd.DataFrame()

    # Ensure 'stock_id' is last column
    if not fund_df.empty:
        cols = [c for c in fund_df.columns if c != "stock_id"] + ["stock_id"]
        fund_df = fund_df[cols]

    print(f"✅ Completed: {len(price_df)} price rows, {len(fund_df)} fundamentals rows.")
    return price_df, fund_df
