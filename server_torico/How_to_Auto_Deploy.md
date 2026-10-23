좋다 🔥
이건 딱 네 상황에서 필요한 **“자동 배포 (CI/CD)” 핵심”**이다.

---

# 🎯 목표

```text
GitHub push → 서버 자동 pull → build → 재시작
```

---

# 🔥 방법 3가지 (난이도 순)

## 1️⃣ 간단 (추천 👍): **git pull + PM2 reload (cron or webhook)**

## 2️⃣ 중급: GitHub Webhook → 서버에서 자동 실행

## 3️⃣ 고급: GitHub Actions → SSH 배포

---

# ✅ 1️⃣ 가장 현실적인 방법 (Webhook 방식)

👉 이거 쓰면 “거의 자동 배포 완성”

---

## 🔧 Step 1: 서버에 deploy script 만들기

```bash
nano ~/deploy.sh
```

```bash
#!/bin/bash

cd ~/apps/myapp || exit

echo "📦 Pulling latest code..."
git pull origin main

echo "📦 Installing..."
npm install

echo "🏗 Building..."
npm run build

echo "♻️ Restarting PM2..."
pm2 reload myapp

echo "✅ Deployment done"
```

권한:

```bash
chmod +x ~/deploy.sh
```

---

## 🔧 Step 2: webhook 서버 만들기 (Node)

```bash
nano webhook.js
```

```js
import express from "express";
import { exec } from "child_process";

const app = express();

app.use(express.json());

app.post("/deploy", (req, res) => {
  console.log("📡 GitHub webhook received");

  exec("bash ~/deploy.sh", (err, stdout, stderr) => {
    if (err) {
      console.error(err);
      return res.status(500).send("Deploy failed");
    }
    console.log(stdout);
    res.send("Deploy success");
  });
});

app.listen(9000, () => {
  console.log("🚀 Webhook server running on 9000");
});
```

---

## 🔧 Step 3: PM2로 webhook 실행

```bash
pm2 start webhook.js --name webhook
pm2 save
```

---

## 🔧 Step 4: GitHub Webhook 설정

👉 GitHub repo → Settings → Webhooks

* URL:

```text
http://your-server-ip:9000/deploy
```

* Content type:

```text
application/json
```

* 이벤트:

```text
Just the push event
```

---

# 🚀 끝

이제:

```text
git push → 자동 deploy
```

---

# ⚠️ 보안 추가 (추천)

webhook 보호:

```js
if (req.headers["x-github-event"] !== "push") {
  return res.status(403).send("Forbidden");
}
```

또는 secret 사용

---

# 🔥 2️⃣ GitHub Actions 방식 (더 깔끔)

`.github/workflows/deploy.yml`

```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: SSH deploy
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: YOUR_SERVER_IP
          username: jwon
          key: ${{ secrets.SSH_KEY }}
          script: |
            cd ~/apps/myapp
            git pull origin main
            npm install
            npm run build
            pm2 reload myapp
```

---

# 💡 너 상황 기준 추천

너 구조:

```text
PM2 + server.ts + coinone_order + cloudflare
```

👉 베스트:

```text
Webhook 방식 + PM2
```

---

# 🔥 한 줄 핵심

```text
GitHub push → webhook → deploy.sh → pm2 reload
```

---

# 🚀 다음 단계 (강추)

지금 너 수준이면 이거까지 가야 된다:

👉 OpenClaw → “deploy latest code” 실행
👉 Discord → “update server”

---

원하면:

👉 **OpenClaw로 자동 배포 명령까지 연결**
👉 **Discord에서 deploy 트리거**

까지 만들어준다 😎
