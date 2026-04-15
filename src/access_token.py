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



def get_auth_token(data):
    host = "https://api.kiwoom.com"
    endpoint = "/oauth2/token"
    url = host + endpoint
    auth_TR = "au10001"
    
    # Header composition
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',  # 요청 데이터의 형식
    }
    
    response = requests.post(url, headers=headers, json=data)
    # response check
    # 4. 응답 상태 코드와 데이터 출력
    print('Code:', response.status_code)
    print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
    print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))  # JSON 응답을 파싱하여 출력
    
    return response.json()
    
    
if __name__ == "__main__":
    # 1. request data
    # body Composition
    params = {
        'grant_type': 'client_credentials',
        'appkey': app_key,
        'secretkey': secret_key
    }
    
    token_info = get_auth_token(data=params)