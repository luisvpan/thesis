export interface RateLimiterOptions {
  windowMs: number;
  maxRequests: number;
}

export interface RateLimitEntry {
  count: number;
  resetTime: number;
}

export class SimpleRateLimiter {
  private windows: Map<string, RateLimitEntry> = new Map();
  private options: RateLimiterOptions;

  constructor(options: RateLimiterOptions = { windowMs: 60000, maxRequests: 100 }) {
    this.options = { ...options };
  }

  check(identifier: string): { allowed: boolean; remaining: number; resetTime: number } {
    const now = Date.now();
    const entry = this.windows.get(identifier);

    if (!entry || now > entry.resetTime) {
      const newEntry: RateLimitEntry = {
        count: 1,
        resetTime: now + this.options.windowMs
      };
      this.windows.set(identifier, newEntry);
      return {
        allowed: true,
        remaining: this.options.maxRequests - 1,
        resetTime: newEntry.resetTime
      };
    }

    if (entry.count >= this.options.maxRequests) {
      return {
        allowed: false,
        remaining: 0,
        resetTime: entry.resetTime
      };
    }

    entry.count++;
    this.windows.set(identifier, entry);
    return {
      allowed: true,
      remaining: this.options.maxRequests - entry.count,
      resetTime: entry.resetTime
    };
  }

  reset(identifier?: string): void {
    if (identifier) {
      this.windows.delete(identifier);
    } else {
      this.windows.clear();
    }
  }

  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.windows.entries()) {
      if (now > entry.resetTime) {
        this.windows.delete(key);
      }
    }
  }
}

export function createRateLimiter(options?: RateLimiterOptions): SimpleRateLimiter {
  return new SimpleRateLimiter(options);
}
