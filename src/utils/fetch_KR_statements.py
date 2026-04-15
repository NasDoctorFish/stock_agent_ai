'''
코드 (reprt_code)	보고서 유형 / 설명
11011	연간사업보고서 (Annual Report) 
engopendart.fss.or.kr

11012	반기보고서 (Semi-annual Report) 
opendart.fss.or.kr

11013	1분기보고서 (First Quarterly Report) 
engopendart.fss.or.kr

11014	3분기보고서 (Third Quarterly Report) 
engopendart.fss.or.kr
'''

'''
colorama example
from colorama import Fore, Style, init
init(autoreset=True)

print(Fore.RED + "에러")
print(Fore.GREEN + "성공")
print(Fore.YELLOW + "경고")
print(Fore.BLUE + "정보")
print(Style.BRIGHT + "강조")
'''
"""
fetch_KR_statements.py

- 관심 corp_code 리스트 기준으로
- 최근 N년간 DART 정기보고서(사업/반기/분기) 목록 수집
- 중복 방지 (rcept_no PK)
- DuckDB 저장까지 포함

실행:
poetry run python -m src.utils.fetch_KR_statements
- 뜻: src 폴더를 패키지 루트로 설정해서 실행해라



"""

# ===============================
# Imports
# ===============================
import os
import json
import re
import time
import requests
import dotenv
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from colorama import Fore, Style, init
import duckdb
from urllib.parse import urljoin, urlparse, parse_qs
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# status logger — use package-relative imports (not src.core.*)
from core.status_logger import StatusLogger
from core.dart_disclosure_analyzer import DartDisclosureAnalyzer

init(autoreset=True)
dotenv.load_dotenv()

# ===============================
# Constants
# ===============================
OPEN_DART_API_KEY = os.getenv("OPEN_DART_API_KEY")
DUCKDB_PATH = "data/dart.duckdb"

logger = StatusLogger("Fetch_KR")

DART_REPORT_CODES = {
    "ANNUAL": "11011",
    "SEMI": "11012",
    "Q1": "11013",
    "Q3": "11014",
}

TARGET_REPRT_CODES = set(DART_REPORT_CODES.values())

DART_MAIN_URL = "https://dart.fss.or.kr/dsaf001/main.do"
DART_VIEWER_BASE = "https://dart.fss.or.kr/report/viewer.do"
DART_BASE = "https://dart.fss.or.kr"
REQUEST_TIMEOUT = (5, 20)
MAX_VIEWER_SECTIONS = 20

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "close",
}


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _build_session()


def _request_get(url: str, params: dict | None = None, referer: str | None = None) -> requests.Response | None:
    headers = dict(REQUEST_HEADERS)
    if referer:
        headers["Referer"] = referer

    for attempt in range(3):
        try:
            return SESSION.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.SSLError as exc:
            logger.warn(Fore.YELLOW + f"[WARN] SSL error on {url}: {exc}")
        except requests.exceptions.RequestException as exc:
            logger.warn(Fore.YELLOW + f"[WARN] Request error on {url}: {exc}")

        time.sleep(0.6 * (attempt + 1))

    return None



# 기업명만 일단 dict로 관리하고 DART에서 corpCode.xml 다운 후 corp_code와 매핑
KR_CORE_CORP_NAMES = {
    "semiconductor": [
        "삼성전자",
        "SK하이닉스",
        "DB하이텍",
        "한미반도체",
        "에스에프에이",
    ],
    "bio": [
        "삼성바이오로직스",
        "셀트리온",
        "에스케이바이오팜",
        "한미약품",
        "유한양행",
    ],
    "defense": [
        "한화에어로스페이스",
        "LIG넥스원",
        "한국항공우주",
        "현대로템",
    ]
}




# ===============================
# Utils
# ===============================
def build_corp_sector_mapping(core_dict: dict) -> pd.DataFrame:
    records = []
    for sector, companies in core_dict.items():
        for corp_name, corp_code in companies.items():
            records.append({
                "corp_code": corp_code,
                "corp_name": corp_name,
                "sector": sector,
                "is_core": True,
            })
    df = pd.DataFrame(records)
    validate_unique_corp_codes(df)
    return df


def validate_unique_corp_codes(df: pd.DataFrame):
    dup = df[df.duplicated("corp_code", keep=False)]
    if not dup.empty:
        raise ValueError(
            Fore.RED + f"Duplicate corp_code detected:\n{dup}"
        )


# ===============================
# DART Fetcher
# ===============================
def _dart_dns_ok() -> bool:
    """Quick DNS pre-check before looping over corp codes."""
    import socket
    try:
        socket.getaddrinfo("opendart.fss.or.kr", 443, socket.AF_INET, socket.SOCK_STREAM)
        return True
    except socket.gaierror as exc:
        logger.warn(f"[DART] DNS resolution failed for opendart.fss.or.kr: {exc}")
        return False


def fetch_dart_filing_list(
    corp_codes: list[str],
    api_key: str,
    years: int = 5,
    target_reports: set[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch DART filing list for the given corp_codes.

    Uses the SESSION object (with retry adapter) instead of bare requests.get().
    Aborts early if DART DNS is unreachable.

    Args:
        corp_codes:     List of 8-digit DART corp codes
        api_key:        OPEN_DART_API_KEY
        years:          Look-back window in years
        target_reports: Set of report codes (e.g. {"11011", "11013"});
                        None = all report types

    Returns:
        DataFrame with columns: corp_code, corp_name, rcept_no, dcm_no,
                                 reprt_code, report_name, report_date
    """
    if not api_key:
        logger.warn("[DART] OPEN_DART_API_KEY is not set — skipping filing fetch")
        return pd.DataFrame()

    if not _dart_dns_ok():
        logger.warn("[DART] Aborting filing fetch — host unreachable")
        return pd.DataFrame()

    url = "https://opendart.fss.or.kr/api/list.json"
    today = datetime.today()

    base_params = {
        "crtfc_key": api_key,
        "bgn_de": (today - relativedelta(years=years)).strftime("%Y%m%d"),
        "end_de": today.strftime("%Y%m%d"),
        "page_count": 100,
    }

    records = []
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    for corp_code in tqdm(corp_codes, desc="Fetching DART filings"):
        for reprt_code in target_reports or [None]:
            params = dict(base_params)
            params["corp_code"] = corp_code
            if reprt_code:
                params["reprt_code"] = reprt_code

            try:
                # Use SESSION (has retry adapter) instead of bare requests.get
                res = SESSION.get(url, params=params, timeout=10)
                res.raise_for_status()
                data = res.json()
                consecutive_errors = 0  # reset on success

                if data.get("status") != "000":
                    logger.warn(
                        Fore.YELLOW + f"[DART][{corp_code}] API error: {data.get('message')}"
                    )
                    continue

                for f in data.get("list", []):
                    records.append({
                        "corp_code": corp_code,
                        "corp_name": f.get("corp_name"),
                        "rcept_no": f.get("rcept_no"),
                        "dcm_no": f.get("dcm_no"),
                        "reprt_code": f.get("reprt_code"),
                        "report_name": f.get("report_nm"),
                        "report_date": f.get("rcept_dt"),
                    })

            except requests.exceptions.ConnectionError as exc:
                consecutive_errors += 1
                logger.warn(Fore.YELLOW + f"[DART][{corp_code}] Connection error: {exc}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.warn("[DART] Too many consecutive errors — aborting fetch")
                    return pd.DataFrame(records)

            except Exception as exc:
                logger.warn(Fore.YELLOW + f"[DART][{corp_code}] {exc}")

    return pd.DataFrame(records)


"""
Viewer.do url 직접 생성
"""

def build_viewer_url(rcept_no: str, dcm_no: str) -> str:
    return (
        f"{DART_VIEWER_BASE}?rcpNo={rcept_no}"
        f"&dcmNo={dcm_no}"
        "&eleId=0&offset=0&length=0&dtd=dart3.xsd"
    )


def build_viewer_url_with_ele(rcept_no: str, dcm_no: str, ele_id: str) -> str:
    return (
        f"{DART_VIEWER_BASE}?rcpNo={rcept_no}"
        f"&dcmNo={dcm_no}"
        f"&eleId={ele_id}"
        "&offset=0&length=0&dtd=dart3.xsd"
    )



# ===============================
# DuckDB Storage
# ===============================
def init_duckdb(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dart_filings (
            corp_code TEXT,
            corp_name TEXT,
            rcept_no TEXT PRIMARY KEY,
            reprt_code TEXT,
            report_name TEXT,
            report_date DATE,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)


def save_to_duckdb(df: pd.DataFrame, db_path: str):
    logger.info("[Fetch KR Statements] saving to duckdb ...")
    if df.empty:
        print(Fore.YELLOW + "No data to save.")
        return

    # convert date into duckdb date formate 
    # YYYYMMDD to YYYY-MM-DD
    df = df.copy()
    df['report_date'] = pd.to_datetime(df['report_date'], format="%Y%m%d", errors="coerce").dt.date

    # df['report_date'] not valid or na then print error and delete
    # ❌ 변환 실패 row 처리
    bad_mask = df["report_date"].isna()
    if bad_mask.any():
        bad_rows = df[bad_mask]
        logger.info(Fore.YELLOW + f"[WARN] Invalid report_date rows dropped: {len(bad_rows)}")
        logger.info(bad_rows[["corp_name", "rcept_no", "report_date"]].head())
        df = df[~bad_mask]

    if df.empty:
        logger.error(Fore.YELLOW + "All rows dropped due to invalid dates.")
        return

    conn = duckdb.connect(db_path)
    init_duckdb(conn)

    conn.register("df_tmp", df)
    conn.execute("""
        INSERT OR IGNORE INTO dart_filings
        SELECT
            corp_code,
            corp_name,
            rcept_no,
            reprt_code,
            report_name,
            CAST(report_date AS DATE),
            CURRENT_TIMESTAMP
        FROM df_tmp
    """)
    conn.close()

    logger.info(Fore.GREEN + f"Saved {len(df)} records into DuckDB")
    
    
"""
Fetching Document Content
"""
import zipfile
import io

"""
documents.xml 파싱해보고 안돼면
viewer.do로 파로 긁기
"""

def fetch_document_structure(api_key: str, rcept_no: str) -> list[dict]:
    url = "https://opendart.fss.or.kr/api/document.xml"
    params = {
        "crtfc_key": api_key,
        "rcept_no": rcept_no
    }

    res = requests.get(url, params=params, timeout=10)

    if not res.ok:
        print(f"[WARN] document.xml HTTP error: {res.status_code}")
        return []

    # ZIP 풀기
    try:
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            xml_name = z.namelist()[0]
            xml_data = z.read(xml_name)
            print(xml_data[:500]) # testing xml data
    except Exception as e:
        print(f"[WARN] ZIP parse failed ({rcept_no}): {e}")
        return []

    # XML 파싱 (여기서 자주 깨짐)
    try:
        root = safe_parse_xml(xml_data)
        if root is None:
            print(f"[WARN] XML unrecoverable ({rcept_no})")
            return []
    except Exception as e:
        print(f"[WARN] XML parse failed ({rcept_no}): {e}")
        return []

    sections = []
    for doc in root.findall(".//document"):
        sections.append({
            "rcept_no": rcept_no,
            "dcm_no": doc.findtext("dcm_no"),
            "title": doc.findtext("title") or "UNKNOWN",
            "ele_id": doc.findtext("ele_id") or "0",
        })

    return sections


"""
safe parse distinguishing <TITLE> or <title>
"""
from lxml import etree

def safe_parse_xml(xml_bytes: bytes):
    parser = etree.XMLParser(
        recover=True,
        huge_tree=True,
        encoding="euc-kr"
    )
    try:
        return etree.fromstring(xml_bytes, parser=parser)
    except Exception as e:
        logger.warn(f"lxml parse failed: {e}")
        return None


"""
Final version, fetch with viewer.do in DART
"""
from bs4 import BeautifulSoup

def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _extract_viewer_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: set[str] = set()

    for a in soup.select("a[href*='viewer.do']"):
        href = a.get("href")
        if href:
            urls.add(urljoin(DART_BASE, href))

    iframe = soup.select_one("iframe[src*='viewer.do']")
    if iframe and iframe.get("src"):
        urls.add(urljoin(DART_BASE, iframe["src"]))

    for match in re.findall(r"viewer\.do\?[^\"'>\s]+", html):
        urls.add(urljoin(DART_BASE, match))

    for match in re.findall(r"viewDoc\('(?P<rcp>\d+)'\s*,\s*'(?P<dcm>\d+)'\s*,\s*'(?P<ele>\d+)'", html):
        urls.add(build_viewer_url_with_ele(match[0], match[1], match[2]))

    return list(urls)


def _choose_best_viewer_url(rcept_no: str, urls: list[str]) -> str | None:
    if not urls:
        return None

    def score(url: str) -> tuple[int, int]:
        query = parse_qs(urlparse(url).query)
        rcp = query.get("rcpNo", [""])[0]
        ele_id = query.get("eleId", [""])[0]
        return (0 if rcp == rcept_no else 1, 0 if ele_id == "0" else 1)

    urls_sorted = sorted(urls, key=score)
    return urls_sorted[0]


def _extract_dcm_no_from_html(html: str) -> str | None:
    match = re.search(r"dcmNo[=:'\"]+(\d+)", html)
    if match:
        return match.group(1)
    match = re.search(r"viewDoc\('(?P<rcp>\d+)'\s*,\s*'(?P<dcm>\d+)'", html)
    if match:
        return match.group("dcm")
    return None


def _fetch_main_html(rcept_no: str) -> str | None:
    res = _request_get(DART_MAIN_URL, params={"rcpNo": rcept_no})
    if not res or not res.ok:
        return None
    return res.content.decode("euc-kr", errors="ignore")


def _resolve_viewer_url(rcept_no: str, dcm_no: str | None = None) -> tuple[str | None, str | None]:
    if dcm_no:
        return build_viewer_url(rcept_no, dcm_no), None

    html = _fetch_main_html(rcept_no)
    if not html:
        return None, None

    urls = _extract_viewer_urls(html)
    best_url = _choose_best_viewer_url(rcept_no, urls)
    if best_url:
        return best_url, html

    dcm_no = _extract_dcm_no_from_html(html)
    if dcm_no:
        return build_viewer_url(rcept_no, dcm_no), html
    return None, html


def _fetch_viewer_text(url: str, referer: str | None = None) -> str:
    res = _request_get(url, referer=referer)
    if not res or not res.ok:
        return ""

    html = res.content.decode("euc-kr", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("body")
    text = body.get_text("\n", strip=True) if body else ""
    if not text:
        text = soup.get_text("\n", strip=True)
    return _normalize_text(text)


def _fetch_all_sections_from_main(rcept_no: str, main_html: str) -> str:
    urls = _extract_viewer_urls(main_html)
    if not urls:
        return ""

    texts: list[str] = []
    referer = f"{DART_MAIN_URL}?rcpNo={rcept_no}"
    for url in urls[:MAX_VIEWER_SECTIONS]:
        section_text = _fetch_viewer_text(url, referer=referer)
        if section_text:
            texts.append(section_text)

    return _normalize_text("\n".join(texts)) if texts else ""


def fetch_viewer_body(rcept_no: str, dcm_no: str | None = None) -> str:
    viewer_url, main_html = _resolve_viewer_url(rcept_no, dcm_no)
    if viewer_url:
        referer = f"{DART_MAIN_URL}?rcpNo={rcept_no}"
        text = _fetch_viewer_text(viewer_url, referer=referer)
        if text:
            return text

    if not main_html:
        main_html = _fetch_main_html(rcept_no)
    if main_html:
        return _fetch_all_sections_from_main(rcept_no, main_html)

    return ""


# ===============================
# Disclosure Analysis + Storage
# ===============================
def init_disclosure_duckdb(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dart_disclosures (
            corp_code TEXT,
            rcept_no TEXT PRIMARY KEY,
            reprt_code TEXT,
            importance_level TEXT,
            tags TEXT,
            title_summary TEXT,
            paragraph_summary TEXT,
            raw_text TEXT
        );
    """)


def save_disclosures_to_duckdb(records: list[dict], db_path: str):
    if not records:
        logger.warn(Fore.YELLOW + "No disclosure documents to save.")
        return

    df = pd.DataFrame(records)
    conn = duckdb.connect(db_path)
    init_disclosure_duckdb(conn)
    conn.register("df_docs", df)
    conn.execute("""
        INSERT OR REPLACE INTO dart_disclosures
        SELECT
            corp_code,
            rcept_no,
            reprt_code,
            importance_level,
            tags,
            title_summary,
            paragraph_summary,
            raw_text
        FROM df_docs
    """)
    conn.close()
    logger.info(Fore.GREEN + f"Saved {len(df)} disclosure documents into DuckDB")


def collect_disclosure_documents(
    df_filings: pd.DataFrame,
    analyzer: DartDisclosureAnalyzer,
) -> list[dict]:
    records: list[dict] = []

    for row in tqdm(df_filings.itertuples(index=False), total=len(df_filings), desc="Scraping DART bodies"):
        rcept_no = getattr(row, "rcept_no", None)
        if not rcept_no:
            continue

        dcm_no = getattr(row, "dcm_no", None)
        if isinstance(dcm_no, float) and pd.isna(dcm_no):
            dcm_no = None
        if isinstance(dcm_no, str) and not dcm_no.strip():
            dcm_no = None

        raw_text = fetch_viewer_body(rcept_no, dcm_no)
        if not raw_text:
            logger.warn(Fore.YELLOW + f"[WARN] Empty body for rcept_no={rcept_no}")
            continue

        analysis = analyzer.analyze(raw_text)
        records.append({
            "corp_code": getattr(row, "corp_code", None),
            "rcept_no": rcept_no,
            "reprt_code": getattr(row, "reprt_code", None),
            "importance_level": analysis["importance_level"],
            "tags": json.dumps(analysis["tags"], ensure_ascii=False),
            "title_summary": analysis["title_summary"],
            "paragraph_summary": analysis["paragraph_summary"],
            "raw_text": raw_text,
        })

    return records


# ===============================
# Financial Statement Fetch
# (structured data for KRStockDataAnalyst)
# ===============================

# DART financial statement API endpoint
_DART_FINSTATE_URL = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

# Account codes for key P&L / BS items
_DART_ACCOUNT_CODES = {
    "revenue":          "ifrs-full_Revenue",
    "operating_income": "ifrs-full_ProfitLossFromOperatingActivities",
    "net_income":       "ifrs-full_ProfitLoss",
    "total_assets":     "ifrs-full_Assets",
    "total_equity":     "ifrs-full_Equity",
}


def fetch_kr_financials(
    corp_code: str,
    api_key: str,
    year: int | None = None,
    reprt_code: str = "11011",  # 11011 = annual report
) -> dict:
    """
    Fetch key financial statement figures for a single Korean company from DART.

    Uses the DART single company financial statement API
    (opendart.fss.or.kr/api/fnlttSinglAcntAll.json).

    Args:
        corp_code:  8-digit DART corporate code
        api_key:    OPEN_DART_API_KEY
        year:       Fiscal year (4-digit int); defaults to last completed year
        reprt_code: "11011" annual | "11012" semi-annual | "11013" Q1 | "11014" Q3

    Returns:
        dict with keys: corp_code, year, revenue, operating_income, net_income,
                        total_assets, total_equity, operating_margin, net_margin
                        (all values in KRW billions; None if unavailable)

    Example:
        data = fetch_kr_financials("00126380", api_key)
        # {"corp_code": "00126380", "revenue": 302231.0, "net_income": 26908.0, ...}
    """
    if not api_key:
        logger.warn("[DART] fetch_kr_financials: OPEN_DART_API_KEY not set")
        return {}

    if not _dart_dns_ok():
        logger.warn("[DART] fetch_kr_financials: host unreachable")
        return {}

    from datetime import datetime as _dt
    if year is None:
        year = _dt.today().year - 1  # last completed fiscal year

    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": "CFS",  # CFS = consolidated financial statements
    }

    try:
        res = SESSION.get(_DART_FINSTATE_URL, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
    except Exception as exc:
        logger.warn(f"[DART] fetch_kr_financials({corp_code}): {exc}")
        return {}

    if data.get("status") != "000":
        logger.warn(
            f"[DART] fetch_kr_financials({corp_code}): "
            f"API error {data.get('status')} — {data.get('message')}"
        )
        return {}

    # Parse the flat list of account items into a lookup dict
    items = data.get("list", [])
    lookup: dict[str, float] = {}
    for item in items:
        acnt_id = item.get("account_id", "")
        thstrm_amount = item.get("thstrm_amount", "")  # current year amount
        if acnt_id and thstrm_amount:
            try:
                # DART amounts are in KRW; convert to billions for readability
                lookup[acnt_id] = float(str(thstrm_amount).replace(",", "")) / 1e9
            except ValueError:
                pass

    rev = lookup.get(_DART_ACCOUNT_CODES["revenue"])
    op_inc = lookup.get(_DART_ACCOUNT_CODES["operating_income"])
    net_inc = lookup.get(_DART_ACCOUNT_CODES["net_income"])
    assets = lookup.get(_DART_ACCOUNT_CODES["total_assets"])
    equity = lookup.get(_DART_ACCOUNT_CODES["total_equity"])

    op_margin = (op_inc / rev) if (rev and op_inc) else None
    net_margin = (net_inc / rev) if (rev and net_inc) else None

    return {
        "corp_code": corp_code,
        "year": year,
        "reprt_code": reprt_code,
        "revenue_bn_krw": rev,
        "operating_income_bn_krw": op_inc,
        "net_income_bn_krw": net_inc,
        "total_assets_bn_krw": assets,
        "total_equity_bn_krw": equity,
        "operating_margin": op_margin,
        "net_margin": net_margin,
    }


def fetch_kr_financials_batch(
    corp_codes: list[str],
    api_key: str,
    year: int | None = None,
    reprt_code: str = "11011",
) -> pd.DataFrame:
    """
    Fetch financial statements for multiple Korean companies.

    Args:
        corp_codes: List of 8-digit DART corp codes
        api_key:    OPEN_DART_API_KEY
        year:       Fiscal year; defaults to last completed year
        reprt_code: Report type code

    Returns:
        DataFrame with one row per company; NaN for unavailable fields.
    """
    if not api_key:
        logger.warn("[DART] fetch_kr_financials_batch: OPEN_DART_API_KEY not set")
        return pd.DataFrame()

    records = []
    for corp_code in tqdm(corp_codes, desc="Fetching KR financials"):
        result = fetch_kr_financials(corp_code, api_key, year, reprt_code)
        if result:
            records.append(result)
        time.sleep(0.3)  # be polite to DART servers

    return pd.DataFrame(records) if records else pd.DataFrame()


# ===============================
# Main
# ===============================
def main():
    # Guard: fail fast if API key is missing
    if not OPEN_DART_API_KEY:
        raise RuntimeError(
            "OPEN_DART_API_KEY is not set in .env — cannot fetch DART data"
        )

    # ===============================
    # Core Corp Registry
    # ===============================
    from utils.corp_code_registry import load_corp_code_table, resolve_corp_codes

    corp_code_df = load_corp_code_table(OPEN_DART_API_KEY)

    records = []
    for sector, names in KR_CORE_CORP_NAMES.items():
        df = resolve_corp_codes(names, corp_code_df)
        df["sector"] = sector
        df["is_core"] = True
        records.append(df)

    df_mapping = pd.concat(records, ignore_index=True)

    print(df_mapping[["corp_code", "corp_name", "stock_code", "sector"]])
    
    
    if not OPEN_DART_API_KEY:
        raise RuntimeError("OPEN_DART_API_KEY not set")

    print(Fore.BLUE + "Building corp registry...")
    corp_codes = df_mapping["corp_code"].unique().tolist()

    print(Fore.BLUE + f"Target corp count: {len(corp_codes)}")

    print(Fore.BLUE + "Fetching DART filings...")
    df_filings = fetch_dart_filing_list(
        corp_codes=corp_codes,
        api_key=OPEN_DART_API_KEY,
        years=5,
        target_reports=TARGET_REPRT_CODES,
    ) # fetch df_filings dataFrame including rcept_no, dcm_no

    print(Fore.BLUE + f"Fetched rows: {len(df_filings)}")
    print(df_filings.head())

    print(Fore.BLUE + "Saving to DuckDB...")
    save_to_duckdb(df_filings, DUCKDB_PATH)

    print(Fore.BLUE + "Loading local LLM analyzer...")
    analyzer = DartDisclosureAnalyzer("models/qwen2.5-1.5b-instruct")

    print(Fore.BLUE + "Scraping disclosure bodies + analyzing...")
    disclosure_records = collect_disclosure_documents(df_filings, analyzer)

    print(Fore.BLUE + "Saving disclosure documents to DuckDB...")
    save_disclosures_to_duckdb(disclosure_records, DUCKDB_PATH)

    print(Style.BRIGHT + Fore.GREEN + "DONE ✅")

    

if __name__ == "__main__":
    print("Main >>>")
    main()

#일단 지금 corpcode별 조회하는것을 빼고, 내가 관심있는 특정 corp_code list만 정리하는 로직이 필요... 아무래도 GPT 5.0 api가 필요할듯 일단 보류

#프롬프트: corpcodes 리스틀르 받으면 각 리스트별로 tqdm해서 최근 공시자료 list를 뽑아옴
#그리고 데이터프레임으로 필수적인 데이터를 정리하고 저장해두고, 이걸 리턴함..
# 추가적인 함수로 duckdb에 바로 저장하는 함수를 만들예정


#이 API 구매를한번해보자
# GPT가 내 입맛에 따라 유효한 기업들 리스트를 자동으로 뽑는다. 그리고 그 리스트를 corp_code 리스트 데이터베이스에 추가한다. 
