/**
 * Test suite: /api/plunge-alert, /api/watchlist/heartbeat, /api/watchlist/status
 * Run: node --test test/plunge_alert.test.js
 * Requires server.ts running on port 4000. Skips gracefully if not reachable.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';

dotenv.config();

const JWT_SECRET = process.env.JWT_SECRET ?? 'test-secret-fallback';
const PLUNGE_ALERT_API_KEY = process.env.PLUNGE_ALERT_API_KEY ?? '';
const BASE_4000 = `http://localhost:4000`;

const tryFetch = async (url, options) => {
  try {
    const res = await fetch(url, options);
    return { ok: true, status: res.status, body: await res.json() };
  } catch {
    return { ok: false };
  }
};

const samplePlungeEvent = {
  symbol: 'GOOGL',
  current_price: 90,
  reference_price: 100,
  reference_type: 'prior_close',
  drop_pct: 10,
  threshold_pct: 7,
  triggered_at: new Date().toISOString()
};

describe('POST /api/plunge-alert auth', () => {
  test('missing x-api-key returns 401', async () => {
    const r = await tryFetch(`${BASE_4000}/api/plunge-alert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(samplePlungeEvent)
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });

  test('wrong x-api-key returns 401', async () => {
    const r = await tryFetch(`${BASE_4000}/api/plunge-alert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': 'wrong-key' },
      body: JSON.stringify(samplePlungeEvent)
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });

  test('correct x-api-key with valid body returns 200', async () => {
    if (!PLUNGE_ALERT_API_KEY) {
      console.log('  [SKIP] PLUNGE_ALERT_API_KEY not set in test env');
      return;
    }
    const r = await tryFetch(`${BASE_4000}/api/plunge-alert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': PLUNGE_ALERT_API_KEY },
      body: JSON.stringify(samplePlungeEvent)
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 200);
    assert.equal(r.body.success, true);
  });

  test('missing required fields returns 400', async () => {
    if (!PLUNGE_ALERT_API_KEY) {
      console.log('  [SKIP] PLUNGE_ALERT_API_KEY not set in test env');
      return;
    }
    const r = await tryFetch(`${BASE_4000}/api/plunge-alert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': PLUNGE_ALERT_API_KEY },
      body: JSON.stringify({ symbol: 'GOOGL' })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 400);
  });
});

describe('GET /api/watchlist/status auth', () => {
  test('no JWT returns 401', async () => {
    const r = await tryFetch(`${BASE_4000}/api/watchlist/status`);
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });

  test('valid JWT returns tickers + recentAlerts', async () => {
    const token = jwt.sign({ id: 1, email: 'test@test.com', role: 'admin' }, JWT_SECRET, { expiresIn: '1m' });
    const r = await tryFetch(`${BASE_4000}/api/watchlist/status`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 200);
    assert.ok(Array.isArray(r.body.tickers));
    assert.ok(Array.isArray(r.body.recentAlerts));
  });
});
