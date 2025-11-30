-- 📊 3개월 수익률, 모멘텀, 변동성
CREATE TABLE features_stock AS
SELECT
    p1.stock_id,
    p1.date,
    (p3.close - p1.close) / p1.close AS fwd_return_3m,
    (p1.close - LAG(p1.close, 60) OVER (PARTITION BY p1.stock_id ORDER BY p1.date)) / LAG(p1.close, 60) OVER (PARTITION BY p1.stock_id ORDER BY p1.date) AS momentum_3m,
    STDDEV(p1.close) OVER (PARTITION BY p1.stock_id ORDER BY p1.date ROWS BETWEEN 60 PRECEDING AND CURRENT ROW) AS volatility_3m
FROM price_daily p1
LEFT JOIN price_daily p3
  ON p1.stock_id = p3.stock_id
  AND p3.date = p1.date + INTERVAL '90 days';

-- 💰 최근 ROE 및 PER 계산
ALTER TABLE features_stock
ADD COLUMN roe_ttm FLOAT,
ADD COLUMN pe_ratio FLOAT;

UPDATE features_stock f
SET
  roe_ttm = (
    SELECT AVG(roe)
    FROM fundamentals_quarterly fq
    WHERE fq.stock_id = f.stock_id
      AND fq.fiscal_date >= f.date - INTERVAL '1 year'
      AND fq.fiscal_date <= f.date
  ),
  pe_ratio = (
    SELECT p1.close / fq.eps
    FROM price_daily p1
    JOIN fundamentals_quarterly fq
      ON fq.stock_id = p1.stock_id
    WHERE p1.stock_id = f.stock_id
      AND p1.date = f.date
    LIMIT 1
  );

-- 📰 감성 점수 7일 평균
UPDATE features_stock f
SET sentiment_avg_7d = (
  SELECT AVG(sentiment_score)
  FROM news_sentiment n
  WHERE n.stock_id = f.stock_id
    AND n.date BETWEEN f.date - INTERVAL '7 days' AND f.date
);
