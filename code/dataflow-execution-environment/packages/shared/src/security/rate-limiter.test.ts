import { describe, it, expect } from "bun:test";
import { SimpleRateLimiter, createRateLimiter } from "@dataflow/shared/security/rate-limiter";

describe("Rate Limiter", () => {
  it("should allow requests under the limit", () => {
    const limiter = createRateLimiter({
      windowMs: 60000,
      maxRequests: 5
    });
    
    const result1 = limiter.check("client1");
    expect(result1.allowed).toBe(true);
    expect(result1.remaining).toBe(4);
    
    const result2 = limiter.check("client1");
    expect(result2.allowed).toBe(true);
    expect(result2.remaining).toBe(3);
    
    const result3 = limiter.check("client1");
    expect(result3.allowed).toBe(true);
    expect(result3.remaining).toBe(2);
  });

  it("should block requests over the limit", () => {
    const limiter = createRateLimiter({
      windowMs: 60000,
      maxRequests: 2
    });
    
    const result1 = limiter.check("client2");
    expect(result1.allowed).toBe(true);
    
    const result2 = limiter.check("client2");
    expect(result2.allowed).toBe(true);
    expect(result2.remaining).toBe(0);
    
    const result3 = limiter.check("client2");
    expect(result3.allowed).toBe(false);
    expect(result3.remaining).toBe(0);
  });

  it("should track different clients independently", () => {
    const limiter = createRateLimiter({
      windowMs: 60000,
      maxRequests: 2
    });
    
    const client1Result1 = limiter.check("client4");
    expect(client1Result1.allowed).toBe(true);
    expect(client1Result1.remaining).toBe(1);
    
    const client1Result2 = limiter.check("client4");
    expect(client1Result2.allowed).toBe(true);
    expect(client1Result2.remaining).toBe(0);
    
    const client1Result3 = limiter.check("client4");
    expect(client1Result3.allowed).toBe(false);
    expect(client1Result3.remaining).toBe(0);
    
    const client2Result1 = limiter.check("client5");
    expect(client2Result1.allowed).toBe(true);
    expect(client2Result1.remaining).toBe(1);
    
    const client2Result2 = limiter.check("client5");
    expect(client2Result2.allowed).toBe(true);
    expect(client2Result2.remaining).toBe(0);
    
    const client2Result3 = limiter.check("client5");
    expect(client2Result3.allowed).toBe(false);
    expect(client2Result3.remaining).toBe(0);
  });

  it("should reset specific client", () => {
    const limiter = createRateLimiter({
      windowMs: 60000,
      maxRequests: 2
    });
    
    const result1 = limiter.check("client6");
    expect(result1.allowed).toBe(true);
    expect(result1.remaining).toBe(1);
    
    limiter.reset("client6");
    
    const result2 = limiter.check("client6");
    expect(result2.allowed).toBe(true);
    expect(result2.remaining).toBe(1);
  });

  it("should reset all clients", () => {
    const limiter = createRateLimiter({
      windowMs: 60000,
      maxRequests: 2
    });
    
    limiter.check("client7");
    limiter.check("client8");
    
    limiter.reset();
    
    const result1 = limiter.check("client7");
    expect(result1.allowed).toBe(true);
    expect(result1.remaining).toBe(1);
    
    const result2 = limiter.check("client8");
    expect(result2.allowed).toBe(true);
    expect(result2.remaining).toBe(1);
  });
});
