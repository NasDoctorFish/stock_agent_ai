






좋아. **맥에서 개발한 Poetry 기반 Python 프로젝트를 서버(리눅스)로 그대로 올려서 실행하는 가장 깔끔한 방식**을 단계별로 정리해줄게.
이 순서대로 하면 100% 문제 없이 “맥 ↔ 서버 동일 환경”이 복제됨.

---

# ✅ **1. 서버(리눅스)에 필요한 것 설치**

### (1) Python 설치

보통 Ubuntu/Debian 기준:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```

만약 Python 3.10/3.11이 필요하면:

```bash
sudo apt install python3.10 python3.10-venv
```

---

### (2) Poetry 설치 (리눅스 공식 방식)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

PATH 등록:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

확인:

```bash
poetry --version
```

---

# ✅ **2. 맥에서 쓰던 프로젝트를 서버로 복사**

가장 깔끔한 방법:

### 🟩 방법 A: GitHub / GitLab / Bitbucket으로 푸시

→ 서버에서 clone만 하면 끝.

```bash
git clone https://github.com/you/stock-ml.git
cd stock-ml
```

### 🟦 방법 B: scp로 직접 전송

맥에서:

```bash
scp -r ~/path/to/stock-ml user@server-ip:/home/user/
```

---

# 🚨 매우 중요: `.venv` 폴더는 **절대 복사하지 않는다**

왜냐면:

* 맥 .venv = ARM64 + macOS용
* 리눅스에서 실행 불가

서버에서 새로 환경을 만드는 것이 정석이다.

---

# ✅ **3. 서버에서 Poetry 환경 생성 + 의존성 설치**

프로젝트 폴더로 이동:

```bash
cd stock-ml
```

### (1) 만약 pyproject.toml이 package-mode=false가 아니라면

추가로 넣어두면 더 편함:

```toml
[tool.poetry]
package-mode = false
```

### (2) Poetry 환경 생성 + 설치:

```bash
poetry install
```

Poetry가 자동으로 `.venv`를 만들고 Linux용 패키지들을 설치함.

---

# 🔥 **4. 서버에서도 Hugging Face 로그인 필요 (중요)**

토큰은 로컬과 서버가 완전히 분리됨.
서버에서도:

```bash
poetry run hf auth login
```

→ 토큰 붙여넣기.

---

# 🧪 **5. 서버에서 프로그램 실행**

예:

```bash
poetry run python3 streamlit_test/pages/3_huggingface.py
```

Streamlit이면:

```bash
poetry run streamlit run app.py
```

---

# 🧱 **6. 서버 RAM/GPU 고려 — 모델 캐시 공유할 공간 생성**

서버에서는 모델 캐시를 한 곳에 모으는 것이 안정적임.

```bash
mkdir -p ~/.cache/huggingface
echo 'export HF_HOME="$HOME/.cache/huggingface"' >> ~/.bashrc
source ~/.bashrc
```

이제 모든 HF 모델은 여기에 저장됨.

---

# 🧩 **7. Mac ↔ Server 환경 완전 동일하게 유지하려면?**

## ✔ pyenv + poetry 조합을 추천

서버에서도 같은 Python 버전 사용 가능:

```bash
curl https://pyenv.run | bash
```

설치 후:

```bash
pyenv install 3.10.13
pyenv local 3.10.13
poetry env use ~/.pyenv/versions/3.10.13/bin/python
```

이러면 맥과 완전 동일 버전의 Python 환경이 구축됨.

---

# 🎯 **최종 요약**

### 서버에 프로젝트를 옮겨서 실행하려면:

1️⃣ 서버에 Python + Poetry 설치
2️⃣ 프로젝트 전체를 서버로 복사
3️⃣ `.venv`는 복사하지 않음
4️⃣ 서버에서 `poetry install`로 새 환경 생성
5️⃣ `hf auth login` 다시 해야 함
6️⃣ `poetry run python`으로 실행

이러면 **맥에서 개발한 코드 = 서버에서 그대로 재현됨.**

---

# 😊 원하면 더 구체적으로 세팅해줄게:

* Ubuntu / Debian / Rocky Linux에 맞춘 완전 자동 설치 스크립트
* 서버에서 Streamlit을 24시간 서비스로 띄우는 systemd 설정
* GPU 서버라면 모델 최적화(TensorRT / bitsandbytes)
* 서버용 Docker 이미지로 패키징

어떤 방식으로 배포하고 싶어?
