'''
poetry run python3 src/utils/fetch_KR_statements.py
'''
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

import OpenDartReader
import dotenv
import os

dotenv.load_dotenv()
api_key = os.getenv("OPEN_DART_API_KEY")

# 1. Get your API key from opendart.fss.or.kr
dart = OpenDartReader(api_key)

# 2. Get Samsung Electronics (005930) Financial Statement for 2023
# '11011' is the code for the Annual Report
financials = dart.finstate(corp='005930', bsns_year='2023', reprt_code='11011')

print(financials)