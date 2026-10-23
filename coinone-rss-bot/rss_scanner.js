#!/usr/bin/env node
/**
 * Coinone RSS Buzz Scanner & Trading Signal Bot
 *
 * Strategy:
 * 1. Polls RSS/news feeds every 5 minutes
 * 2. Counts coin mentions across recent articles (buzz score)
 * 3. If a coin's buzz score exceeds threshold AND sentiment is positive → emit BUY signal
 * 4. Calculates "win ratio" and "position size %" based on buzz intensity
 * 5. Sends signal to your server_torico /api/signal endpoint (or logs if dry-run)
 */

import fetch from 'node-fetch';
import crypto from 'node:crypto';
import { XMLParser } from 'fast-xml-parser';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '../.env') });

// ─── Config ───────────────────────────────────────────────────────────────────
const DRY_RUN = process.env.DRY_RUN !== 'false'; // default: dry run (safe)
const SIGNAL_SERVER = process.env.SIGNAL_SERVER || 'http://localhost:4000';
const SIGNAL_API_KEY = process.env.SIGNAL_API_KEY || '';
const COINONE_ACCESS_KEY = process.env.COINONE_ACCESS_KEY?.trim() || '';
const COINONE_SECRET = process.env.COINONE_SECRET?.trim() || '';
const DISCORD_WEBHOOK = process.env.DISCORD_WEBHOOK?.trim() || '';

const SCAN_INTERVAL_MS = 5 * 60 * 1000;   // 5 minutes
const BUZZ_THRESHOLD = 5;                  // min mentions to trigger signal
const MAX_POSITION_PCT = 10;               // max 10% of portfolio per trade
const COOLDOWN_MS = 30 * 60 * 1000;        // 30 min cooldown per coin after signal

// ─── RSS Feeds ─────────────────────────────────────────────────────────────────
const RSS_FEEDS = [
  // Korean crypto news
  'https://www.coindesk.com/arc/outboundfeeds/rss/',
  'https://cointelegraph.com/rss',
  'https://cryptonews.com/news/feed/',
  'https://decrypt.co/feed',
  // newsapi fallback (uses your NEWS_API_KEY)
];

// ─── Coin keyword map ──────────────────────────────────────────────────────────
// Maps coin ticker (Coinone target_currency) → keywords to search in news
const COIN_KEYWORDS = {
  BTC:  ['bitcoin', 'btc', '비트코인'],
  ETH:  ['ethereum', 'eth', '이더리움'],
  XRP:  ['ripple', 'xrp', '리플'],
  SOL:  ['solana', 'sol', '솔라나'],
  DOGE: ['dogecoin', 'doge', '도지코인'],
  ADA:  ['cardano', 'ada', '카르다노'],
  AVAX: ['avalanche', 'avax', '아발란체'],
  MATIC:['polygon', 'matic', '폴리곤'],
  LINK: ['chainlink', 'link', '체인링크'],
  DOT:  ['polkadot', 'dot', '폴카닷'],
};

// ─── Sentiment keywords ────────────────────────────────────────────────────────
const BULLISH_WORDS = [
  'surge', 'rally', 'bullish', 'breakout', 'moon', 'pump', 'all-time high', 'ath',
  'soar', 'skyrocket', 'adoption', 'partnership', 'institutional', 'etf', 'approve',
  '급등', '상승', '돌파', '최고가', '매수', '강세'
];
const BEARISH_WORDS = [
  'crash', 'dump', 'bearish', 'plunge', 'ban', 'hack', 'scam', 'fraud', 'collapse',
  'regulation', 'sec', 'lawsuit', 'arrest', '급락', '하락', '규제', '매도', '약세'
];

// ─── State ─────────────────────────────────────────────────────────────────────
const lastSignalAt = {}; // { coin: timestamp }
const parser = new XMLParser({ ignoreAttributes: false });

// ─── Helpers ───────────────────────────────────────────────────────────────────
function log(msg) {
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19);
  console.log(`[${ts}] ${msg}`);
}

async function fetchRSS(url) {
  try {
    const resp = await fetch(url, { timeout: 10000, headers: { 'User-Agent': 'CoinoneRSSBot/1.0' } });
    if (!resp.ok) return [];
    const xml = await resp.text();
    const data = parser.parse(xml);
    const items = data?.rss?.channel?.item || data?.feed?.entry || [];
    return Array.isArray(items) ? items : [items];
  } catch (e) {
    log(`RSS fetch error (${url}): ${e.message}`);
    return [];
  }
}

async function fetchNewsAPI(coin) {
  const key = process.env.NEWS_API_KEY;
  if (!key) return [];
  const query = COIN_KEYWORDS[coin]?.[0] || coin.toLowerCase();
  try {
    const resp = await fetch(
      `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&language=en&sortBy=publishedAt&pageSize=20&apiKey=${key}`
    );
    const data = await resp.json();
    return (data.articles || []).map(a => ({ title: a.title, description: a.description }));
  } catch (e) {
    log(`NewsAPI error: ${e.message}`);
    return [];
  }
}

function scoreArticle(text, coin) {
  const lower = text.toLowerCase();
  const keywords = COIN_KEYWORDS[coin] || [coin.toLowerCase()];
  const mentions = keywords.reduce((n, kw) => {
    const matches = lower.split(kw.toLowerCase()).length - 1;
    return n + matches;
  }, 0);
  if (mentions === 0) return null;

  let sentiment = 0;
  BULLISH_WORDS.forEach(w => { if (lower.includes(w)) sentiment++; });
  BEARISH_WORDS.forEach(w => { if (lower.includes(w)) sentiment--; });

  return { mentions, sentiment };
}

/**
 * Win ratio model:
 *  - Based on buzz score (mentions) and net sentiment
 *  - Scale: buzz 5-9 → ~55% win, 10-19 → ~65%, 20+ → ~72%
 *  - Sentiment bonus: +1% per net positive sentiment point (capped at +8%)
 *  - These are heuristic estimates, NOT financial guarantees
 */
function calcSignalParams(buzzScore, netSentiment) {
  let baseWinRatio;
  if (buzzScore >= 20) baseWinRatio = 72;
  else if (buzzScore >= 10) baseWinRatio = 60 + Math.floor((buzzScore - 10) / 2);
  else baseWinRatio = 50 + buzzScore; // 55-59% for 5-9

  const sentimentBonus = Math.min(Math.max(netSentiment, 0) * 1.5, 8);
  const winRatio = Math.min(baseWinRatio + sentimentBonus, 80); // cap at 80%

  // Position size: scale 2%–10% based on win ratio
  const positionPct = Math.min(((winRatio - 50) / 30) * MAX_POSITION_PCT, MAX_POSITION_PCT);

  return {
    winRatioPct: Math.round(winRatio * 10) / 10,
    positionPct: Math.round(positionPct * 10) / 10
  };
}

// ─── Coinone direct order (bypass local server if needed) ──────────────────────
function signCoinone(payload) {
  const raw = JSON.stringify(payload);
  const base64Payload = Buffer.from(raw).toString('base64');
  const signature = crypto
    .createHmac('sha512', COINONE_SECRET)
    .update(base64Payload)
    .digest('hex');
  return { base64Payload, signature };
}

async function placeCoinoneOrder(coin, side, qty) {
  const payload = {
    access_token: COINONE_ACCESS_KEY,
    nonce: Date.now().toString(),
    side,
    type: 'MARKET',
    qty: String(qty),
    target_currency: coin,
    quote_currency: 'KRW'
  };
  const { base64Payload, signature } = signCoinone(payload);
  const resp = await fetch('https://api.coinone.co.kr/v2.1/order', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-COINONE-PAYLOAD': base64Payload,
      'X-COINONE-SIGNATURE': signature
    },
    body: JSON.stringify(payload)
  });
  return resp.json();
}

// ─── Discord notification ──────────────────────────────────────────────────────
async function notifyDiscord(embed) {
  if (!DISCORD_WEBHOOK) return;
  try {
    await fetch(DISCORD_WEBHOOK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embeds: [embed] })
    });
  } catch (e) {
    log(`Discord notify error: ${e.message}`);
  }
}

// ─── Main scan ─────────────────────────────────────────────────────────────────
async function scan() {
  log('🔍 Starting RSS buzz scan...');

  // Fetch all RSS articles
  const allArticles = [];
  for (const feed of RSS_FEEDS) {
    const items = await fetchRSS(feed);
    allArticles.push(...items);
  }
  log(`Fetched ${allArticles.length} RSS articles across ${RSS_FEEDS.length} feeds`);

  const results = {};

  for (const coin of Object.keys(COIN_KEYWORDS)) {
    let totalMentions = 0;
    let totalSentiment = 0;
    const matchedTitles = [];

    // Score RSS articles
    for (const item of allArticles) {
      const text = [item.title, item.description, item.summary, item['content:encoded']]
        .filter(Boolean).join(' ');
      const score = scoreArticle(text, coin);
      if (score) {
        totalMentions += score.mentions;
        totalSentiment += score.sentiment;
        if (matchedTitles.length < 3 && item.title) matchedTitles.push(item.title);
      }
    }

    // Augment with NewsAPI if low buzz
    if (totalMentions < BUZZ_THRESHOLD) {
      const newsItems = await fetchNewsAPI(coin);
      for (const item of newsItems) {
        const text = [item.title, item.description].filter(Boolean).join(' ');
        const score = scoreArticle(text, coin);
        if (score) {
          totalMentions += score.mentions;
          totalSentiment += score.sentiment;
        }
      }
    }

    results[coin] = { mentions: totalMentions, sentiment: totalSentiment, matchedTitles };
  }

  // Print summary
  log('─── Buzz Summary ──────────────────────────────');
  for (const [coin, r] of Object.entries(results)) {
    log(`  ${coin.padEnd(6)}: mentions=${r.mentions}, sentiment=${r.sentiment >= 0 ? '+' : ''}${r.sentiment}`);
  }
  log('───────────────────────────────────────────────');

  // Emit signals for coins above threshold
  for (const [coin, r] of Object.entries(results)) {
    if (r.mentions < BUZZ_THRESHOLD) continue;
    if (r.sentiment <= 0) continue; // only bullish signals

    const now = Date.now();
    if (lastSignalAt[coin] && (now - lastSignalAt[coin]) < COOLDOWN_MS) {
      log(`⏳ ${coin}: cooldown active, skipping`);
      continue;
    }

    const { winRatioPct, positionPct } = calcSignalParams(r.mentions, r.sentiment);

    log(`🚀 SIGNAL: BUY ${coin} | buzz=${r.mentions} | sentiment=+${r.sentiment} | win_ratio=${winRatioPct}% | position=${positionPct}%`);
    if (r.matchedTitles.length) {
      log(`   Top headlines: ${r.matchedTitles[0]}`);
    }

    lastSignalAt[coin] = now;

    const embed = {
      title: `📡 RSS Signal: BUY ${coin}`,
      color: 0x00ff88,
      fields: [
        { name: 'Buzz Score', value: `${r.mentions} mentions`, inline: true },
        { name: 'Sentiment', value: `+${r.sentiment}`, inline: true },
        { name: 'Estimated Win Ratio', value: `**${winRatioPct}%**`, inline: true },
        { name: 'Suggested Position', value: `**${positionPct}%** of portfolio`, inline: true },
        { name: 'Top Headline', value: r.matchedTitles[0] || '(none)', inline: false },
      ],
      footer: { text: `Coinone RSS Bot • ${DRY_RUN ? 'DRY RUN' : 'LIVE'} • ${new Date().toISOString()}` }
    };

    await notifyDiscord(embed);

    if (!DRY_RUN) {
      // Send to local signal server (which handles Coinone auth)
      try {
        const resp = await fetch(`${SIGNAL_SERVER}/api/signal`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(SIGNAL_API_KEY ? { 'X-API-KEY': SIGNAL_API_KEY } : {})
          },
          body: JSON.stringify({
            side: 'BUY',
            type: 'MARKET',
            qty: positionPct, // server can normalize this to actual qty
            targetCurrency: coin,
            quoteCurrency: 'KRW'
          })
        });
        const data = await resp.json();
        log(`   Order result: ${JSON.stringify(data)}`);
      } catch (e) {
        log(`   Order error: ${e.message}`);
      }
    } else {
      log(`   [DRY RUN] Order NOT placed. Set DRY_RUN=false to go live.`);
    }
  }

  log('✅ Scan complete.\n');
}

// ─── Entry ─────────────────────────────────────────────────────────────────────
log(`🤖 Coinone RSS Bot starting | DRY_RUN=${DRY_RUN} | interval=${SCAN_INTERVAL_MS / 60000}m`);
scan();
setInterval(scan, SCAN_INTERVAL_MS);
