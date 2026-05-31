# Investment Intelligence System

## Project Structure

```text
project/
├── data/
│   ├── raw/
│   └── processed/
│
├── sql/
│   ├── schema.sql
│   └── features.sql
│
├── src/
│   ├── core/
│   │   ├── db_handler.py
│   │   ├── db_updater.py
│   │   ├── decision_maker.py
│   │   └── investor.py
│   │
│   ├── analysts/
│   │   ├── base_analyst.py
│   │   ├── news_analyst.py
│   │   ├── macro_analyst.py
│   │   ├── fundamental_analyst.py
│   │   └── technical_analyst.py
│   │
│   └── config/
│       └── settings.yaml
│
├── reports/
└── README.md
```

# Data Layer

## price_daily

```csv
date,open,high,low,close,volume,stock_id
```

## fundamentals_quarterly

```csv
date,total_revenue,operating_income,net_income,eps,stock_id
```

## news_sentiment

```csv
date,headline,sentiment_score,stock_id
```

## macro_index

```csv
date,index_name,value
```

# System Architecture

```text
Investor
    │
    ▼
DecisionMaker
    │
    ├── NewsAnalyst
    ├── MacroAnalyst
    ├── FundamentalAnalyst
    └── TechnicalAnalyst
    │
    ▼
DBHandler
```

# Component Responsibilities

## Investor

* Execute investment cycle
* Request analysis
* Receive final decision

## DecisionMaker

* Combine analyst outputs
* Generate Buy / Hold / Sell decision

## Analysts

### NewsAnalyst

* News sentiment analysis

### MacroAnalyst

* Interest rates
* Economic indicators

### FundamentalAnalyst

* Revenue
* Earnings
* ROE
* Valuation

### TechnicalAnalyst

* Price trend
* Volume
* Technical indicators

## DBUpdater

* Collect new data
* Update database

## DBHandler

* Query data
* Insert data
* Manage storage

# Data Flow

```text
External Data
      │
      ▼
DBUpdater
      │
      ▼
Database
      │
      ▼
Analysts
      │
      ▼
DecisionMaker
      │
      ▼
Investor
```

# Goal

Generate investment decisions using:

* Market Data
* Fundamental Data
* News Data
* Macro Data

Output:

```text
BUY
HOLD
SELL
```
