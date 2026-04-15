import requests
import json
from dotenv import load_dotenv
import os

# .env 로드
load_dotenv()

# 환경변수 가져오기
app_key = os.getenv("KIWOOM_APP_KEY")
secret_key = os.getenv("KIWOOM_SECRET_KEY")

print(app_key)
print(secret_key)


#url
host = 'https://api.kiwoom.com' # 실전투자 서버
endpoint = '/api/dostk/ordr' # 매수 주문 API
url = host + endpoint


# setting
token = 'aaaaaaaaa'
cont_yn = 'y'
next_key = 'dddddd'

# Header composition
headers = {
    'Content-Type': 'application/json;charset=UTF-8',  # 요청 데이터의 형식
    'authorization': f'Bearer {token}',  # 인증 토큰
    'cont-yn': cont_yn,  # 연속조회 여부
    'next-key': next_key,  # 연속조회 키
    'api-id': 'kt10000',  # 요청하는 TR 이름
}

# body Composition
data = {
    'dmst_stex_tp': 'KRX',  # 거래소 구분 (예: KRX)
    'stk_cd': '039490',  # 주문할 종목코드 (키움증권)
    'ord_qty': '1',  # 주문 수량
    'ord_uv': '',  # 주문 단가 (시장가 주문은 빈 값)
    'trde_tp': '3',  # 매매 구분 (3: 시장가 주문)
    'cond_uv': '',  # 조건 단가 (해당 사항 없으면 빈 값)
}

response = requests.post(url, headers=headers, json=data)

print('Code:', response.status_code)  # HTTP 상태 코드 출력
print('Header:', json.dumps(dict(response.headers), indent=4, ensure_ascii=False))  # 응답 헤더 출력
print('\n\nBETTER VIEW\n')
for k, v in response.headers.items():
    print(f"{k}: {v}")
print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))  # 응답 바디 출력



# --------------------------___________________--------------------------

https://api.kiwoom.com

def get_auth_token(self) -> JSON:
    host = "https://api.kiwoom.com"
    url = "/oauth2/token"
    auth_TR = "au10001"
    
    # Header composition
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',  # 요청 데이터의 형식
        'authorization': f'Bearer {token}',  # 인증 토큰
        'cont-yn': cont_yn,  # 연속조회 여부
        'next-key': next_key,  # 연속조회 키
        'api-id': 'kt10000',  # 요청하는 TR 이름
    }

    # body Composition
    data = {
        'grant_type': "",
        'app_key': '',
        'secretkey': ''
    }
            