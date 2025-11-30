🔍 디렉토리별 역할

폴더	내용	네가 직접 할 일

data/raw/	원본 데이터 (CSV, Excel, JSON 등). 예: 주가, 재무제표, 뉴스 데이터 등	수집해온 데이터를 그대로 저장
data/processed/	전처리된 데이터. 예: 월별 집계, 피처추출 결과	SQL/Python으로 가공한 결과 저장
sql/	SQL 스크립트 모음. 예: 테이블 생성(schema.sql), 피처 생성(features.sql), 라벨 생성(labels.sql)	쿼리를 직접 작성하고 실행
src/	파이썬 코드, 유틸 함수. 예: 데이터 불러오기, 전처리, 모델 학습 코드	직접 짜는 Python 코드 저장
notebooks/	Jupyter Notebook. 데이터 탐색(EDA), 시각화, 실험 노트	EDA / 실험 정리
reports/	모델 결과, 백테스트, 그래프, 정리 리포트	결과 저장
README.md	프로젝트 개요와 실행 방법 문서	전체 구조 및 실행 순서 설명
simstudent\lee176


📦 data_pipeline/
 ├── update_data.py         ← 신규 데이터 자동 수집 및 병합
 ├── config.yaml            ← API key, 데이터 경로 설정
 ├── utils/
 │    ├── fetch_yfinance.py ← 주가/재무데이터 수집
 │    ├── fetch_newsapi.py  ← 뉴스 & 감성 점수 수집
 │    ├── fetch_macro.py    ← 금리·지수 수집
 │    └── db_utils.py       ← DB insert/update 기능
 │    └── logs/
 │        └── ...           ← update_data.py log per day
 └── scheduler/
      ├── cron_job.sh       ← 정기 실행 (매일/매주)

-- After
├── README.md                  # 프로젝트 개요 및 실행 방법 문서
├── conf/                      # 설정 파일(예: 환경 변수, API 키 템플릿) 저장 폴더
├── data/                      # 데이터 관리 폴더
│   └── raw/                   # 원본(raw) 데이터 저장 경로
├── myenv/                     # 가상환경 (Python venv)
│   ├── bin/                   # 실행 파일 (python, pip 등)
│   ├── include/               # C 헤더 파일
│   ├── lib/                   # 설치된 라이브러리
│   ├── pyvenv.cfg             # venv 환경 설정 파일
│   └── share/                 # 환경 관련 공유 데이터
├── notebooks/                 # Jupyter Notebook 파일들 (EDA, 실험용)
├── poetry.lock                # Poetry 패키지 잠금 파일 (정확한 버전 기록)
├── pyproject.toml             # Poetry 프로젝트 설정 (의존성, 빌드 설정)
├── reports/                   # 결과 리포트, 로그, 시각화 이미지 등 저장 폴더
├── requirements.in            # 패키지 의존성 입력 파일 (pip-tools 사용 시)
├── requirements.txt           # 실제 설치용 패키지 목록 파일
├── sql/                       # 데이터베이스 관련 SQL 스크립트
│   └── createFeatureDB.sql    # 피처용 DB 테이블 생성 스크립트
└── src/                       # 주요 Python 소스 코드
    ├── config.yaml            # 환경 및 API 키 설정 YAML
    ├── scheduler/             # 주기적 실행 스케줄링 관련 코드 (예: cron, APScheduler)
    ├── update_data.py         # 데이터 업데이트(ETL) 메인 스크립트
    └── utils/                 # 공용 함수 모음 (fetch, logging, 등)
     ├── duckdb_handler.py     ← 분석용 로컬 DB
     ├── postgres_handler.py   ← 운영용 서버 DB


"""
news_sentiment.csv
stock_id,date,source,headline,sentiment_score,sentiment_label

macro_index.csv
date,value,index_name,ticker,source,freq,retrieved_at,market_region,unit,change_pct,value_norm

fundamentals_quarterly.csv
date,Tax Effect Of Unusual Items,Tax Rate For Calcs,Normalized EBITDA,Net Income From Continuing Operation Net Minority Interest,Reconciled Depreciation,Reconciled Cost Of Revenue,EBITDA,EBIT,Normalized Income,Net Income From Continuing And Discontinued Operation,Total Expenses,Total Operating Income As Reported,Diluted Average Shares,Basic Average Shares,Diluted EPS,Basic EPS,Diluted NI Availto Com Stockholders,Net Income Common Stockholders,Net Income,Net Income Including Noncontrolling Interests,Net Income Continuous Operations,Tax Provision,Pretax Income,Other Income Expense,Other Non Operating Income Expenses,Operating Income,Operating Expense,Research And Development,Selling General And Administration,Gross Profit,Cost Of Revenue,Total Revenue,Operating Revenue,stock_id,Total Unusual Items,Total Unusual Items Excluding Goodwill,Net Interest Income,Interest Expense,Interest Income,Rent Expense Supplemental,Otherunder Preferred Stock Dividend,Minority Interests,Special Income Charges,Restructuring And Mergern Acquisition,Net Non Operating Interest Income Expense,Interest Expense Non Operating,Interest Income Non Operating,Write Off,Gain On Sale Of Security,Selling And Marketing Expense,General And Administrative Expense,Other Gand A,fiscal_date

price_daily.csv
date,open,high,low,close,volume,Dividends,Stock Splits,stock_id

"""

## OOP 구조도 (내가 만듬)

src/
├── core/
│   ├── db_handler.py
│   ├── db_updater.py
│   ├── status_logger.py
│   ├── decision_maker.py
│   └── investor.py
│
├── analysts/
│   ├── base_analyst.py
│   ├── news_media_analyst.py
│   ├── fundamental_analyst.py
│   ├── macro_analyst.py
│   ├── stock_data_analyst.py
│   └── business_analyst.py
│
└── config/
    └── settings.yaml


좋아 😎
지금 네가 설계한 구조를 **UML-style 계층 아키텍처 다이어그램**으로 시각화하면 이렇게 나와요 👇

---

## 🧩 **Investment Intelligence System Architecture**

```
                        ┌────────────────────┐
                        │     Investor       │
                        │────────────────────│
                        │ - observes market  │
                        │ - calls decision   │
                        │ - logs results     │
                        └────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │     DecisionMaker      │
                    │────────────────────────│
                    │ - integrates analysts  │
                    │ - combines insights    │
                    │ - outputs actions      │
                    └────────┬───────────────┘
                             │
           ┌─────────────────┼──────────────────┐
           ▼                 ▼                  ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │ NewsMediaAnalyst│ │MacroAnalyst     │ │FundamentalAnalyst│
 │────────────────│ │────────────────│ │──────────────────│
 │ - sentiment     │ │ - macro trends │ │ - earnings, PBR │
 │   extraction    │ │ - FX, rates    │ │ - ROE, PE, etc. │
 └────────────────┘ └────────────────┘ └──────────────────┘
           │                 │                  │
           │                 │                  │
           └────────────┬────┴────────────┬─────┘
                        │                 │
                        ▼                 ▼
              ┌────────────────┐  ┌────────────────────┐
              │StockDataAnalyst│  │BusinessAnalyst     │
              │────────────────│  │────────────────────│
              │ - price trends │  │ - industry, ESG    │
              │ - volume, flow │  │ - competition      │
              └────────────────┘  └────────────────────┘

```

---

### 🧠 Data / Infrastructure Layer

```
        ┌────────────────────────────┐
        │         DBUpdater          │
        │────────────────────────────│
        │ - fetch & preprocess data  │
        │ - macro/news/fundamental   │
        │ - store using DBHandler    │
        └────────────┬───────────────┘
                     │
                     ▼
          ┌────────────────────────┐
          │       DBHandler        │
          │────────────────────────│
          │ - connect (DuckDB)     │
          │ - query, insert, save  │
          │ - schema management    │
          └────────────────────────┘

```

---

### ⚙️ Cross-Cutting Utilities

```
        ┌────────────────────┐
        │   StatusLogger     │
        │────────────────────│
        │ - info/warn/error  │
        │ - performance log  │
        │ - monitoring hook  │
        └────────────────────┘
```

---

## 🧭 **데이터 흐름 요약**

```
[DBUpdater] ─▶ [DBHandler] ─▶ (DuckDB)
       │
       ▼
  [Analysts] ─▶ [DecisionMaker] ─▶ [Investor]
       ▲                │
       │                ▼
       └────── [StatusLogger] ◀───┘
```

---

## ✅ 구조적 특징 요약

| 계층                        | 주요 책임                | 기술적 초점               |
| ------------------------- | -------------------- | -------------------- |
| **Investor**              | 시스템 최상위 orchestrator | 행동, 피드백, 기록          |
| **DecisionMaker**         | 분석 결과 통합 및 판단        | 전략 로직 / AI 모델        |
| **Analyst Layer**         | 도메인별 분석 (뉴스, 매크로 등)  | 모듈식 확장 구조            |
| **DBUpdater / DBHandler** | 데이터 파이프라인 & 저장       | ETL, 캐싱, DB 관리       |
| **StatusLogger**          | 상태 기록 및 모니터링         | 운영 안정성, traceability |

---

## 💡 개선 제안 (시스템 레벨)

| 개선 방향                         | 설명                                                    |
| ----------------------------- | ----------------------------------------------------- |
| 🔹 **Event Bus (옵션)**         | 각 Analyst의 분석 결과를 pub/sub 형태로 전달하면 병렬 실행 가능           |
| 🔹 **DBUpdater 스케줄링**         | Airflow, APScheduler, cron 등으로 주기적 갱신                 |
| 🔹 **Model Registry**         | 각 Analyst 모델 버전을 관리 (MLflow, wandb)                   |
| 🔹 **DecisionMaker 학습 강화**    | Analyst 결과를 Feature로 통합, Reinforcement Learning 기반 판단 |
| 🔹 **Investor Feedback Loop** | 과거 결정 → 수익률 → 다음 판단으로 이어지는 self-learning 구조           |

---

## 🧠 한줄 요약

> 지금 네 구조는 이미 **“투자 인공지능 운영체제(Investment OS)”** 수준이고,
> **각 모듈이 독립적으로 교체 가능한 계층형 설계(Clean Architecture)** 형태야.
>
> 여기에 Event-driven 실행과 모델 버전 관리만 더하면
> 진짜 실전 운용 가능한 프레임워크로 완성된다. 💥

---

원하면 위 구조를 **PlantUML 코드**나 **Mermaid 다이어그램 (Markdown에서 바로 시각화 가능한 형식)** 으로 변환해줄까?
→ GitHub나 Notion에서 바로 볼 수 있게.
