import express from 'express';
import type { NextFunction, Request, Response } from 'express';
import axios from 'axios';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
import * as QRCode from 'qrcode';
import { generateSecret, generateURI, verifySync } from 'otplib';
import rateLimit from 'express-rate-limit';
import crypto, { webcrypto } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sendDiscordAlert, sendDiscordSummary, type PlungeSummaryPayload } from './discord.js';
import { LoginLockoutStore } from './lockout.js';

dotenv.config();

if (!globalThis.crypto) {
  globalThis.crypto = webcrypto as unknown as Crypto;
}

const PORT = Number(process.env.PORT ?? 4000);
const JWT_SECRET = process.env.JWT_SECRET ?? '';
const ACCESS_TOKEN = process.env.ACCESS_TOKEN ?? '';
const SECRET_KEY = process.env.SECRET_KEY ?? '';
const SIGNAL_API_KEY = process.env.SIGNAL_API_KEY ?? '';
const PLUNGE_ALERT_API_KEY = process.env.PLUNGE_ALERT_API_KEY ?? '';

if (!JWT_SECRET) {
  throw new Error('환경변수 JWT_SECRET가 없습니다.');
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const staticDir = path.join(__dirname, process.env.NODE_ENV === 'production' ? '../src' : '.');
const logsDir = path.join(__dirname, '../logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

const dataDir = path.join(__dirname, '../data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}
const alertsFilePath = path.join(dataDir, 'alerts.json');
const watchlistStatusFilePath = path.join(dataDir, 'watchlist_status.json');
const MAX_ALERT_HISTORY = 500;

const getLogFilePath = () => {
  const today = new Date().toISOString().split('T')[0];
  return path.join(logsDir, `${today}.log`);
};

const writeLog = (level: string, message: string, meta?: Record<string, unknown>) => {
  const entry = JSON.stringify({ timestamp: new Date().toISOString(), level, message, ...meta }) + '\n';
  fs.appendFileSync(getLogFilePath(), entry);
  if (level === 'ERROR') console.error(entry.trim());
  else console.log(entry.trim());
};

const app = express();
// Required so req.ip reflects the real client IP (via X-Forwarded-For) instead of the
// Cloudflare tunnel's local socket address — needed for per-IP lockout to work correctly.
app.set('trust proxy', 1);

interface User {
  id: number;
  email: string;
  name: string;
  role: 'admin' | 'user';
  otpSecret?: string;
}

interface JwtPayload {
  id: number;
  email: string;
  role: User['role'];
}

type OrderSide = 'BUY' | 'SELL';
type OrderType = 'LIMIT' | 'MARKET' | 'STOP_LIMIT';

interface OrderRequest {
  side: OrderSide;
  type: OrderType;
  price?: number;
  qty: number;
  triggerPrice?: number;
  targetCurrency?: string;
  quoteCurrency?: string;
}

const users: User[] = [
  { id: 1, name: 'LEE JUWON', role: 'admin', email: 'juwonlee3465@gmail.com' }
];

const issueAccessToken = (user: User) => {
  return jwt.sign({ id: user.id, email: user.email, role: user.role }, JWT_SECRET, {
    expiresIn: '5m'
  });
};

const authenticateJWT = (req: Request, res: Response, next: NextFunction) => {
  const authHeader = req.headers.authorization ?? '';
  const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
  
  writeLog('INFO', 'JWT auth attempt', { method: req.method, path: req.path, hasToken: !!token });

  if (!token) {
    return res.status(401).json({ success: false, message: 'JWT 토큰이 필요합니다.' });
  }

  try {
    jwt.verify(token, JWT_SECRET) as JwtPayload;
    return next();
  } catch (error) {
    return res.status(401).json({ success: false, message: 'JWT 인증 실패' });
  }
};

const authorizeSignal = (req: Request, res: Response, next: NextFunction) => {
  const apiKey = req.headers['x-api-key'];
  const authHeader = req.headers.authorization ?? '';
  const bearerToken = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';

  if (SIGNAL_API_KEY && apiKey === SIGNAL_API_KEY) {
    return next();
  }

  if (!bearerToken) {
    return res.status(401).json({ success: false, message: 'Signal 인증이 필요합니다.' });
  }

  try {
    jwt.verify(bearerToken, JWT_SECRET) as JwtPayload;
    return next();
  } catch (error) {
    return res.status(401).json({ success: false, message: 'Signal JWT 인증 실패' });
  }
};

// Internal-only: called by the Python plunge poller, never by the browser dashboard.
// Deliberately a separate key from SIGNAL_API_KEY so a leaked alert key cannot place trades.
const authorizeAlertKey = (req: Request, res: Response, next: NextFunction) => {
  const apiKey = req.headers['x-api-key'];
  if (!PLUNGE_ALERT_API_KEY || apiKey !== PLUNGE_ALERT_API_KEY) {
    return res.status(401).json({ success: false, message: 'Alert 인증이 필요합니다.' });
  }
  return next();
};

const loginLockout = new LoginLockoutStore();

const otpRateLimiter = rateLimit({
  windowMs: 60_000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false
});

interface PlungeAlertEvent {
  symbol: string;
  current_price: number;
  reference_price: number;
  reference_type: 'n_day_high' | 'prior_close';
  drop_pct: number;
  threshold_pct: number;
  triggered_at: string;
}

interface WatchlistHeartbeat {
  symbol: string;
  current_price: number;
  last_checked: string;
}

const readJsonFile = <T>(filePath: string, fallback: T): T => {
  if (!fs.existsSync(filePath)) {
    return fallback;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8')) as T;
  } catch {
    return fallback;
  }
};

const appendAlertHistory = (event: PlungeAlertEvent) => {
  const history = readJsonFile<PlungeAlertEvent[]>(alertsFilePath, []);
  history.push(event);
  const trimmed = history.slice(-MAX_ALERT_HISTORY);
  fs.writeFileSync(alertsFilePath, JSON.stringify(trimmed, null, 2));
  return trimmed;
};

const updateWatchlistStatus = (heartbeat: WatchlistHeartbeat) => {
  const status = readJsonFile<Record<string, WatchlistHeartbeat>>(watchlistStatusFilePath, {});
  status[heartbeat.symbol] = heartbeat;
  fs.writeFileSync(watchlistStatusFilePath, JSON.stringify(status, null, 2));
  return status;
};

const normalizeOrder = (payload: OrderRequest) => {
  const targetCurrency = payload.targetCurrency ?? 'BTC';
  const quoteCurrency = payload.quoteCurrency ?? 'KRW';

  if (!payload.side || !payload.type || !payload.qty) {
    throw new Error('주문 필수 값이 없습니다.');
  }

  if (payload.type !== 'MARKET' && !payload.price) {
    throw new Error('LIMIT 또는 STOP_LIMIT 주문은 price가 필요합니다.');
  }

  if (payload.type === 'STOP_LIMIT' && !payload.triggerPrice) {
    throw new Error('STOP_LIMIT 주문은 triggerPrice가 필요합니다.');
  }

  return {
    ...payload,
    targetCurrency,
    quoteCurrency
  };
};

class CoinoneClient {
  private readonly accessToken: string;
  private readonly secretKey: string;
  private readonly http = axios.create({
    baseURL: 'https://api.coinone.co.kr',
    timeout: 15000
  });

  constructor(accessToken: string, secretKey: string) {
    this.accessToken = accessToken;
    this.secretKey = secretKey;
  }

  private signPayload(payload: Record<string, unknown>) {
    const raw = JSON.stringify(payload);
    const base64Payload = Buffer.from(raw).toString('base64');
    const signature = crypto.createHmac('sha512', this.secretKey).update(base64Payload).digest('hex');

    return {
      'X-COINONE-PAYLOAD': base64Payload,
      'X-COINONE-SIGNATURE': signature
    };
  }

  private async postPrivate<T>(endpoint: string, payload: Record<string, unknown>) {
    const body = {
      ...payload,
      access_token: this.accessToken,
      nonce: Date.now().toString()
    };

    const headers = this.signPayload(body);

    const response = await this.http.post<T>(endpoint, body, {
      headers: {
        'Content-Type': 'application/json',
        ...headers
      }
    });

    return response.data;
  }

  async placeOrder(order: OrderRequest) {
    const normalized = normalizeOrder(order);
    const payload: Record<string, unknown> = {
      side: normalized.side,
      type: normalized.type,
      price: normalized.type === 'MARKET' ? undefined : normalized.price,
      qty: normalized.qty,
      trigger_price: normalized.type === 'STOP_LIMIT' ? normalized.triggerPrice : undefined,
      target_currency: normalized.targetCurrency,
      quote_currency: normalized.quoteCurrency
    };

    Object.keys(payload).forEach((key) => {
      if (payload[key] === undefined) {
        delete payload[key];
      }
    });

    return this.postPrivate('/v2.1/order', payload);
  }
}

const coinoneClient = ACCESS_TOKEN && SECRET_KEY ? new CoinoneClient(ACCESS_TOKEN, SECRET_KEY) : null;

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

app.use((req: Request, res: Response, next: NextFunction) => {
  const start = Date.now();
  res.on('finish', () => {
    writeLog('INFO', 'request', {
      method: req.method,
      path: req.path,
      status: res.statusCode,
      ms: Date.now() - start
    });
  });
  next();
});

app.use((req: Request, res: Response, next: NextFunction) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-API-KEY');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }
  return next();
});

app.use(express.static(staticDir));

app.get('/', (req: Request, res: Response) => {
  res.sendFile(path.join(staticDir, 'index.html'));
});

app.get('/api/users/:id/setup-otp', otpRateLimiter, async (req: Request, res: Response) => {
  if (loginLockout.isLocked(req.ip ?? '')) {
    return res.status(429).json({ success: false, message: '너무 많은 시도. 잠시 후 다시 시도하세요.' });
  }

  const user = users.find((entry) => entry.id === Number(req.params.id));
  if (!user) {
    return res.status(404).json({ success: false, message: '유저 없음' });
  }

  const secret = generateSecret();
  user.otpSecret = secret;

  const otpauth = generateURI({
    issuer: 'Coinone Home Server',
    label: user.email,
    secret
  });

  const qrImageUrl = await QRCode.toDataURL(otpauth);

  return res.json({
    success: true,
    secret,
    qrCode: qrImageUrl
  });
});

app.post('/api/verify-otp', otpRateLimiter, (req: Request, res: Response) => {
  const ip = req.ip ?? '';
  if (loginLockout.isLocked(ip)) {
    return res.status(429).json({ success: false, message: '너무 많은 시도. 잠시 후 다시 시도하세요.' });
  }

  const { userId, token } = req.body as { userId: number; token: string };
  const user = users.find((entry) => entry.id === Number(userId));

  if (!user || !user.otpSecret) {
    loginLockout.recordFailure(ip);
    return res.status(400).json({ success: false, message: 'OTP 설정이 필요합니다.' });
  }

  const isValid = verifySync({ token, secret: user.otpSecret });
  if (!isValid) {
    loginLockout.recordFailure(ip);
    return res.status(401).json({ success: false, message: '번호가 틀렸습니다.' });
  }

  loginLockout.recordSuccess(ip);
  return res.json({
    success: true,
    message: '인증 성공!',
    accessToken: issueAccessToken(user)
  });
});

app.get('/api/ticker', async (req: Request, res: Response) => {
  try {
    const response = await axios.get('https://api.coinone.co.kr/public/v2/ticker_new/KRW/BTC');
    return res.json(response.data);
  } catch (error) {
    return res.status(502).json({
      success: false,
      message: '티커 프록시 실패',
      error: (error as Error).message
    });
  }
});

app.post('/api/order', authenticateJWT, async (req: Request, res: Response) => {
  if (!coinoneClient) {
    return res.status(500).json({
      success: false,
      message: 'ACCESS_TOKEN 또는 SECRET_KEY가 없습니다.'
    });
  }

  try {
    const result = await coinoneClient.placeOrder(req.body as OrderRequest);
    return res.json({ success: true, data: result });
  } catch (error) {
    return res.status(400).json({
      success: false,
      message: (error as Error).message
    });
  }
});

app.post('/api/signal', authorizeSignal, async (req: Request, res: Response) => {
  if (!coinoneClient) {
    return res.status(500).json({
      success: false,
      message: 'ACCESS_TOKEN 또는 SECRET_KEY가 없습니다.'
    });
  }

  try {
    const result = await coinoneClient.placeOrder(req.body as OrderRequest);
    return res.json({ success: true, data: result });
  } catch (error) {
    return res.status(400).json({
      success: false,
      message: (error as Error).message
    });
  }
});

app.post('/api/plunge-alert', authorizeAlertKey, async (req: Request, res: Response) => {
  const event = req.body as PlungeAlertEvent;
  if (!event.symbol || typeof event.drop_pct !== 'number') {
    return res.status(400).json({ success: false, message: '필수 필드가 없습니다.' });
  }

  appendAlertHistory(event);
  await sendDiscordAlert(event, writeLog);

  writeLog('INFO', 'Plunge alert received', { symbol: event.symbol, drop_pct: event.drop_pct });
  return res.json({ success: true });
});

app.post('/api/plunge-summary', authorizeAlertKey, async (req: Request, res: Response) => {
  const summary = req.body as PlungeSummaryPayload;
  if (!Array.isArray(summary.symbols) || summary.symbols.length === 0) {
    return res.status(400).json({ success: false, message: '필수 필드가 없습니다.' });
  }

  await sendDiscordSummary(summary, writeLog);

  writeLog('INFO', 'Plunge summary received', { total_events: summary.total_events, symbols: summary.symbols.length });
  return res.json({ success: true });
});

app.post('/api/watchlist/heartbeat', authorizeAlertKey, (req: Request, res: Response) => {
  const heartbeat = req.body as WatchlistHeartbeat;
  if (!heartbeat.symbol) {
    return res.status(400).json({ success: false, message: '필수 필드가 없습니다.' });
  }

  updateWatchlistStatus(heartbeat);
  return res.json({ success: true });
});

app.get('/api/watchlist/status', authenticateJWT, (req: Request, res: Response) => {
  const tickers = readJsonFile<Record<string, WatchlistHeartbeat>>(watchlistStatusFilePath, {});
  const recentAlerts = readJsonFile<PlungeAlertEvent[]>(alertsFilePath, []).slice(-20).reverse();

  return res.json({
    success: true,
    tickers: Object.values(tickers),
    recentAlerts
  });
});

app.use((req: Request, res: Response) => {
  res.status(404).json({ success: false, message: '요청한 리소스를 찾을 수 없습니다.' });
});

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  writeLog('ERROR', err.message, { path: req.path, stack: err.stack });
  res.status(500).json({ success: false, message: '서버 에러가 발생했습니다', error: err.message });
});

app.listen(PORT, () => {
  writeLog('INFO', `Coinone Home Server running at http://localhost:${PORT}`);
});
