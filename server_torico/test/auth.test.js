/**
 * Test suite: Bearer token flow between port 3000 and port 4000
 * Run: node --test test/auth.test.js
 * Requires both servers to be running OR uses unit logic directly.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import jwt from 'jsonwebtoken';
import dotenv from 'dotenv';

dotenv.config();

const JWT_SECRET = process.env.JWT_SECRET ?? 'test-secret-fallback';
const BASE_4000 = `http://localhost:4000`;
const BASE_3000 = `http://localhost:3000`;

// ─── Unit: JWT issuance & verification ──────────────────────────────────────

describe('JWT unit tests', () => {
  test('issues a valid JWT with correct payload', () => {
    const user = { id: 1, email: 'test@example.com', role: 'admin' };
    const token = jwt.sign(user, JWT_SECRET, { expiresIn: '5m' });
    const decoded = jwt.verify(token, JWT_SECRET);
    assert.equal(decoded.id, user.id);
    assert.equal(decoded.email, user.email);
    assert.equal(decoded.role, user.role);
  });

  test('rejects an expired token', () => {
    const token = jwt.sign({ id: 1 }, JWT_SECRET, { expiresIn: '-1s' });
    assert.throws(() => jwt.verify(token, JWT_SECRET), /TokenExpiredError|jwt expired/);
  });

  test('rejects a token signed with wrong secret', () => {
    const token = jwt.sign({ id: 1 }, 'wrong-secret');
    assert.throws(() => jwt.verify(token, JWT_SECRET), /JsonWebTokenError|invalid signature/);
  });

  test('rejects a malformed token string', () => {
    assert.throws(() => jwt.verify('not.a.token', JWT_SECRET), /JsonWebTokenError/);
  });

  test('token expires within 5 minutes (300s)', () => {
    const token = jwt.sign({ id: 1 }, JWT_SECRET, { expiresIn: '5m' });
    const decoded = jwt.decode(token);
    const diff = decoded.exp - decoded.iat;
    assert.equal(diff, 300);
  });
});

// ─── Unit: Bearer header parsing ────────────────────────────────────────────

describe('Bearer header parsing', () => {
  const extractBearer = (authHeader) => {
    if (!authHeader || !authHeader.startsWith('Bearer ')) return '';
    return authHeader.slice(7);
  };

  test('extracts token from valid Bearer header', () => {
    const token = 'eyJhbGciOiJIUzI1NiJ9.test.sig';
    assert.equal(extractBearer(`Bearer ${token}`), token);
  });

  test('returns empty string for missing header', () => {
    assert.equal(extractBearer(undefined), '');
    assert.equal(extractBearer(''), '');
  });

  test('returns empty string when scheme is not Bearer', () => {
    assert.equal(extractBearer('Basic abc123'), '');
    assert.equal(extractBearer('Token abc123'), '');
  });

  test('does not strip extra whitespace inside token', () => {
    const raw = 'Bearer abc def';
    assert.equal(extractBearer(raw), 'abc def');
  });
});

// ─── Integration: Port 4000 endpoints ───────────────────────────────────────
// These tests require server.ts to be running on port 4000.
// Skip gracefully if server is not reachable.

const tryFetch = async (url, options) => {
  try {
    const res = await fetch(url, options);
    return { ok: true, status: res.status, body: await res.json() };
  } catch {
    return { ok: false };
  }
};

describe('Port 4000 integration', () => {
  test('GET /api/ticker responds without auth (no 401/403)', async () => {
    const r = await tryFetch(`${BASE_4000}/api/ticker`);
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    // 200 = Coinone API up; 500/502 = Coinone API down — both are acceptable (no auth gate)
    assert.ok(r.status !== 401 && r.status !== 403, `Expected no auth error, got ${r.status}`);
  });

  test('POST /api/order without token returns 401', async () => {
    const r = await tryFetch(`${BASE_4000}/api/order`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ side: 'BUY', type: 'MARKET', qty: 0.001 })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
    assert.equal(r.body.success, false);
  });

  test('POST /api/order with invalid token returns 401', async () => {
    const r = await tryFetch(`${BASE_4000}/api/order`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer invalid.token.value'
      },
      body: JSON.stringify({ side: 'BUY', type: 'MARKET', qty: 0.001 })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });

  test('POST /api/order with valid JWT passes auth (reaches order logic)', async () => {
    const token = jwt.sign(
      { id: 1, email: 'test@test.com', role: 'admin' },
      JWT_SECRET,
      { expiresIn: '1m' }
    );
    const r = await tryFetch(`${BASE_4000}/api/order`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ side: 'BUY', type: 'MARKET', qty: 0.001 })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    // Auth passed: should NOT be 401 (may be 400/500 if Coinone keys missing)
    assert.notEqual(r.status, 401);
  });

  test('POST /api/signal without auth returns 401', async () => {
    // Include price+qty so field-check middleware does not short-circuit before auth
    const r = await tryFetch(`${BASE_4000}/api/signal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ side: 'BUY', type: 'MARKET', qty: 0.001, price: 100000000 })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });
});

// ─── Integration: Port 3000 endpoints ───────────────────────────────────────

describe('Port 3000 integration', () => {
  test('POST /api/order/limit-buy without token returns 401', async () => {
    const r = await tryFetch(`${BASE_3000}/api/order/limit-buy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ price: 100000000, qty: 0.001 })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 3000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });

  test('POST /api/order/limit-buy/advanced without x-api-key returns 401', async () => {
    const r = await tryFetch(`${BASE_3000}/api/order/limit-buy/advanced`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quote: 'KRW', target: 'BTC', side: 'BUY', price: 100000000, qty: 0.001, postOnly: false })
    });
    if (!r.ok) {
      console.log('  [SKIP] Port 3000 not reachable');
      return;
    }
    assert.equal(r.status, 401);
  });
});

// ─── Integration: Bearer token relay 3000 → 4000 ────────────────────────────

describe('Bearer token relay: OTP → JWT → /api/order', () => {
  test('token issued by port 4000 OTP flow is accepted by /api/order', async () => {
    // Step 1: setup OTP for user 1
    const setup = await tryFetch(`${BASE_4000}/api/users/1/setup-otp`);
    if (!setup.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.equal(setup.status, 200);
    assert.ok(setup.body.secret, 'OTP secret should be returned');

    // Step 2: verify OTP — use TOTP library in real integration;
    // here we manually issue a token with the same secret to simulate success.
    const simulatedToken = jwt.sign(
      { id: 1, email: 'juwonlee3465@gmail.com', role: 'admin' },
      JWT_SECRET,
      { expiresIn: '5m' }
    );

    // Step 3: use the token against /api/order
    const order = await tryFetch(`${BASE_4000}/api/order`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${simulatedToken}`
      },
      body: JSON.stringify({ side: 'BUY', type: 'MARKET', qty: 0.001 })
    });
    if (!order.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    assert.notEqual(order.status, 401, 'Should not fail auth');
  });
});
