# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Node/TypeScript, ESM (`"type": "module"`), Express 5.

- Dev (hot reload, runs TS directly via tsx): `npm run dev` or `./manage.sh dev`
- Build: `npm run build` (`tsc` → `dist/`) or `./manage.sh build`
- Start built output: `npm start` (`node dist/server.js`) or `./manage.sh start`
- Start the separate order server: `npm run start:order` (`node src/coinone_order.js`)
- Docker: `./manage.sh docker` (`docker compose up --build`; builds, runs on port 4000)
- All tests: `npm test` (`node --test --test-concurrency=1 test/*.test.js`)
- Single test file: `node --test test/auth.test.js` (also `lockout.test.js`, `plunge_alert.test.js`)

## Architecture

Two separate Express processes, normally both run under pm2 (see `execute_4000.sh`):

- **`src/server.ts` (port 4000, env `PORT`)** — the main API gateway. Handles: JWT auth (`authenticateJWT` middleware, 5-minute tokens), OTP enrollment/verification (`enrollment.ts`, `otplib`), per-IP login lockout (`lockout.ts`'s `LoginLockoutStore` — relies on `app.set('trust proxy', 1)` since real traffic arrives through the Cloudflare tunnel, not directly), Coinone order placement via an in-process `CoinoneClient` (`/api/order`, `/api/signal`), the plunge-alert receiver consumed by `Feb_08_2026`'s monitor (`/api/plunge-alert`, `/api/plunge-summary`, `/api/watchlist/heartbeat`, all gated by `authorizeAlertKey`), and Discord relaying (`discord.ts`: `sendDiscordAlert`/`sendDiscordSummary`). Structured JSON logs written per-day to `logs/YYYY-MM-DD.log` via `writeLog`. Alert/watchlist state persisted to `data/alerts.json` / `data/watchlist_status.json`.
- **`src/coinone_order.js` (port 3000, env `ORDER_PORT`)** — a separate standalone Express app that also places Coinone orders, sharing `JWT_SECRET` with `server.ts` (comment in the file: "must match the port-3000 server"). Its relationship to `server.ts`'s own in-process `CoinoneClient`/`/api/order` is not obvious from the code alone — check which one is actually wired to the live trading path before changing either.

## Deployment

- Public access is a Cloudflare Quick Tunnel pointed at `localhost:4000`, started by the **root** repo's `start_tunnel.sh` (not anything in this directory).
- `How_to_Auto_Deploy.md` documents a pull-based deploy: a `~/deploy.sh` on the server does `git pull origin main && npm install && npm run build && pm2 reload`. The root repo's `pull_git.sh` clones/hard-resets this project into `~/Stock_ML/server_torico_v02` — a separate checkout from whatever copy you're editing. Local edits don't auto-deploy.
- `Dockerfile`/`docker-compose.yml` are an alternate deploy path (build + run on port 4000, `.env` supplied via `env_file`), not obviously the one in active use alongside pm2 — verify which is live before assuming either.
