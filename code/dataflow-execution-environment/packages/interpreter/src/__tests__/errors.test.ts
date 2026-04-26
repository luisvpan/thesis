import { describe, test, expect } from "bun:test";
import { execute } from "../index";

describe("Error handling", () => {
  test("detects circular dependencies", async () => {
    const result = await execute(`
      source x = 1;
      transform a = sum(b, x);
      transform b = sum(a, x);
      sink result = a;
    `);

    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("detects undefined references", async () => {
    const result = await execute(`
      transform a = sum(undefined_var, 1);
      sink result = a;
    `);

    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("division by zero throws error", async () => {
    const result = await execute(`
      source a = 10;
      source b = 0;
      transform bad = divide(a, b);
      sink result = bad;
    `);

    expect(result.errors.length).toBeGreaterThan(0);
  });
});
