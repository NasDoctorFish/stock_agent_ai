interface IpState {
  fails: number;
  lockedUntil: number | null;
}

// Progressive lockout: failures -> lockout duration once that threshold is crossed.
const BACKOFF_SCHEDULE: Array<{ fails: number; lockoutMs: number }> = [
  { fails: 20, lockoutMs: 24 * 60 * 60 * 1000 },
  { fails: 15, lockoutMs: 30 * 60 * 1000 },
  { fails: 10, lockoutMs: 5 * 60 * 1000 },
  { fails: 5, lockoutMs: 60 * 1000 }
];

export class LoginLockoutStore {
  private readonly state = new Map<string, IpState>();

  isLocked(ip: string): boolean {
    const entry = this.state.get(ip);
    if (!entry?.lockedUntil) {
      return false;
    }
    if (Date.now() >= entry.lockedUntil) {
      entry.lockedUntil = null;
      return false;
    }
    return true;
  }

  recordFailure(ip: string): void {
    const entry = this.state.get(ip) ?? { fails: 0, lockedUntil: null };
    entry.fails += 1;

    const tier = BACKOFF_SCHEDULE.find((t) => entry.fails >= t.fails);
    if (tier) {
      entry.lockedUntil = Date.now() + tier.lockoutMs;
    }

    this.state.set(ip, entry);
  }

  recordSuccess(ip: string): void {
    this.state.delete(ip);
  }
}
