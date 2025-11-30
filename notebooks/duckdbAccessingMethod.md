좋은 질문이야 💪
DuckDB는 **SQLite처럼 내장 SQL로 메타데이터를 조회**할 수 있어서,
간단한 명령어로 테이블 목록, 스키마, 컬럼 정보 등을 바로 확인할 수 있어.

---

## 🧾 **DuckDB 기본 명령어 요약**

| 목적              | 명령어 (SQL)                          | 설명                     |
| --------------- | ---------------------------------- | ---------------------- |
| ✅ 전체 테이블 목록     | `SHOW TABLES;`                     | 현재 데이터베이스의 모든 테이블 이름   |
| ✅ 특정 스키마 내 테이블  | `SHOW TABLES FROM main;`           | `main` 스키마만 보기         |
| ✅ 특정 테이블 구조 보기  | `DESCRIBE table_name;`             | 각 컬럼명, 타입, null 허용 여부  |
| ✅ 모든 스키마 보기     | `SHOW SCHEMAS;`                    | 스키마 구조 확인              |
| ✅ 현재 연결 정보      | `PRAGMA database_list;`            | 연결된 DB 파일 위치 등         |
| ✅ 현재 테이블 컬럼 리스트 | `PRAGMA table_info('table_name');` | `DESCRIBE`와 유사하지만 더 상세 |
| ✅ 현재 세션 설정 보기   | `PRAGMA settings;`                 | DuckDB 설정값 전부 확인       |

---

## 🧠 **`DBHandler`에서 직접 실행하기**

너의 `DBHandler`는 이미 `.execute(query, fetch=True)`로
결과를 DataFrame 형태로 반환할 수 있으니까, 이렇게 쓰면 돼 👇

```python
# ✅ 전체 테이블 목록 보기
df = db.execute("SHOW TABLES;", fetch=True)
print(df)
```

예시 출력:

```
       name
0  fundamentals_q_full
1  macro_index_full
2  news_sentiment
3  price_daily_full
```

---

## 🧩 **테이블 구조 확인**

```python
df = db.execute("DESCRIBE price_daily_full;", fetch=True)
print(df)
```

출력 예시:

```
       column_name column_type null
0             date         DATE  YES
1             open       DOUBLE  YES
2            close       DOUBLE  YES
3           volume       BIGINT  YES
4           ticker      VARCHAR  YES
```

---

## ⚙️ **직접 편하게 쓰려면 헬퍼 함수 추가 가능**

(원하면 `DBHandler` 안에 아래처럼 하나 추가해도 돼 👇)

```python
def show_tables(self):
    """현재 DB에 존재하는 모든 테이블 목록 반환"""
    return self.execute("SHOW TABLES;", fetch=True)
```

사용 예:

```python
print(db.show_tables())
```

---

## 🔍 **그 외 유용한 명령**

| 기능         | SQL 예시                                    |
| ---------- | ----------------------------------------- |
| 전체 레코드 수   | `SELECT COUNT(*) FROM price_daily_full;`  |
| 임의 5행 보기   | `SELECT * FROM price_daily_full LIMIT 5;` |
| 테이블 삭제     | `DROP TABLE IF EXISTS price_daily_full;`  |
| 메모리 캐시 플러시 | `PRAGMA memory_limit='4GB';` (제한 변경)      |
| DB 백업      | `EXPORT DATABASE '/backup/path';`         |

---

요약하자면 👇

* `SHOW TABLES;` → 목록
* `DESCRIBE table;` → 구조
* `PRAGMA table_info('table');` → 세부 컬럼
* 전부 `db.execute(sql, fetch=True)`로 바로 실행 가능

---

원하면 내가 `DBHandler`에
`show_tables()`, `describe_table(name)`, `preview_table(name, n=5)`
3개 유틸 함수 통합해서 넣은 버전 만들어줄까?
