"""
corp_code_registry.py

- DART corpCode.xml 다운로드
- 기업명 → corp_code 매핑
- 상장 여부 포함
"""

import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
from colorama import Fore, init

init(autoreset=True)

DART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"


def load_corp_code_table(api_key: str) -> pd.DataFrame:
    """
    DART corpCode.xml 다운로드 후 DataFrame 반환
    """
    params = {"crtfc_key": api_key}

    print(Fore.BLUE + "Downloading corpCode.xml ...")
    res = requests.get(DART_CORP_CODE_URL, params=params, timeout=15)
    res.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        xml_name = z.namelist()[0]
        xml_data = z.read(xml_name)

    root = ET.fromstring(xml_data)

    records = []
    for corp in root.findall("list"):
        records.append({
            "corp_code": corp.findtext("corp_code"),
            "corp_name": corp.findtext("corp_name"),
            "stock_code": corp.findtext("stock_code"),
            "modify_date": corp.findtext("modify_date"),
            "is_listed": corp.findtext("stock_code") not in (None, "", " "),
        })

    df = pd.DataFrame(records)

    print(Fore.GREEN + f"Loaded {len(df)} corp codes")
    return df


def resolve_corp_codes(
    corp_names: list[str],
    corp_code_df: pd.DataFrame,
    strict: bool = True
) -> pd.DataFrame:
    records = []

    for name in corp_names:
        m = corp_code_df[corp_code_df["corp_name"] == name]

        if m.empty:
            msg = f"[NOT FOUND] {name}"
            if strict:
                raise ValueError(msg)
            else:
                print(msg)
                continue

        # ✅ 중복일 경우: 상장 법인 우선
        if len(m) > 1:
            listed = m[m["is_listed"] == True]

            if len(listed) == 1:
                m = listed
            else:
                raise ValueError(
                    f"[AMBIGUOUS NAME] {name}\n{m}"
                )

        records.append(m.iloc[0].to_dict())

    return pd.DataFrame(records)

