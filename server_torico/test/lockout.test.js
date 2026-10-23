/**
 * Test suite: progressive login lockout on /api/verify-otp
 * Run: node --test test/lockout.test.js
 * Requires server.ts running on port 4000. Skips gracefully if not reachable.
 * Note: lockout state is in-memory on the live server and keyed by this test
 * runner's IP — running this suite will actually trip the lockout for that IP
 * until the lockout window (or a server restart) clears it.
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

const BASE_4000 = `http://localhost:4000`;

const tryVerifyOtp = async (token) => {
  try {
    const res = await fetch(`${BASE_4000}/api/verify-otp`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: 1, token })
    });
    return { ok: true, status: res.status };
  } catch {
    return { ok: false };
  }
};

describe('Progressive lockout on /api/verify-otp', () => {
  test('repeated wrong OTP codes degrade from 401 to 429', async () => {
    const probe = await tryVerifyOtp('000000');
    if (!probe.ok) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }

    const statuses = [probe.status];
    for (let i = 0; i < 6; i += 1) {
      const r = await tryVerifyOtp('000000');
      statuses.push(r.status);
    }

    assert.ok(statuses.includes(429), `Expected a 429 after repeated failures, got: ${statuses.join(',')}`);

    // Once locked, even a well-formed request should stay blocked until the window expires.
    const stillLocked = await tryVerifyOtp('123456');
    assert.equal(stillLocked.status, 429);
  });

  test('setup-otp is also rate-limited/lockout-protected (regression check for prior unauth hole)', async () => {
    const r = await fetch(`${BASE_4000}/api/users/1/setup-otp`).catch(() => null);
    if (!r) {
      console.log('  [SKIP] Port 4000 not reachable');
      return;
    }
    // With the same IP already locked out by the previous test, this must not
    // silently succeed and leak a fresh OTP secret.
    assert.equal(r.status, 429);
  });
});
