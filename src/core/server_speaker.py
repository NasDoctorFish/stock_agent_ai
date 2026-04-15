"""
What's necessary

1. **server_speaker → order_ai_server**: 주문 요청 전달 및 주식 스코어 결과 전달
2. **order_ai_server → proxy_server**: 분석 리포트 생성 후 주문 JSON과 타임라인 전송
3. **proxy_server → order_ai_server**: 주문 실행 후 상태 결과 반환
4. **order_ai_server → user**: 주문 상태 리포트 업데이트 및 사용자 알림

"""
"""
juwonlee3465  [3:09 PM]
Server Endpoint
https://railroad-approve-august-base.trycloudflare.com/  => over_due...

외부 노출 IP를 쓰려면 어쩔 수 없는 듯...
"""