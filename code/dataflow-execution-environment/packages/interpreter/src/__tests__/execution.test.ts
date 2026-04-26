import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { execute } from "../index";
import type { RationalValue } from "../runtime/types";

describe("Independent executions", () => {
  test("re-executes with modified source values", async () => {
    // First execution
    const result1 = await execute(`
      source x = 10;
      transform doubled = multiply(x, 2);
      sink result = doubled;
    `);
    expect((result1.results.get("result") as RationalValue).value.equals(new Fraction(20))).toBe(true);

    // Second execution with modified source value
    const result2 = await execute(`
      source x = 50;
      transform doubled = multiply(x, 2);
      sink result = doubled;
    `);
    expect((result2.results.get("result") as RationalValue).value.equals(new Fraction(100))).toBe(true);
  });

  test("re-executes with added statements", async () => {
    // First execution - simple
    const result1 = await execute(`
      source a = 5;
      sink result = a;
    `);
    expect((result1.results.get("result") as RationalValue).value.equals(new Fraction(5))).toBe(true);

    // Second execution - added transform
    const result2 = await execute(`
      source a = 5;
      source b = 3;
      transform total = sum(a, b);
      sink result = total;
    `);
    expect((result2.results.get("result") as RationalValue).value.equals(new Fraction(8))).toBe(true);
  });

  test("re-executes with removed statements", async () => {
    // First execution - complex
    const result1 = await execute(`
      source a = 5;
      source b = 3;
      transform total = sum(a, b);
      sink result = total;
    `);
    expect((result1.results.get("result") as RationalValue).value.equals(new Fraction(8))).toBe(true);

    // Second execution - simplified (removed b and transform)
    const result2 = await execute(`
      source a = 5;
      sink result = a;
    `);
    expect((result2.results.get("result") as RationalValue).value.equals(new Fraction(5))).toBe(true);
  });

  test("re-executes with changed operations", async () => {
    // First execution - sum
    const result1 = await execute(`
      source a = 10;
      source b = 2;
      transform calc = sum(a, b);
      sink result = calc;
    `);
    expect((result1.results.get("result") as RationalValue).value.equals(new Fraction(12))).toBe(true);

    // Second execution - changed to multiply
    const result2 = await execute(`
      source a = 10;
      source b = 2;
      transform calc = multiply(a, b);
      sink result = calc;
    `);
    expect((result2.results.get("result") as RationalValue).value.equals(new Fraction(20))).toBe(true);
  });

  test("executions are independent (no state leakage)", async () => {
    // Execute program A
    const resultA = await execute(`
      source x = 100;
      sink result = x;
    `);

    // Execute completely different program B
    const resultB = await execute(`
      source y = 1;
      sink result = y;
    `);

    // Verify no interference
    expect((resultA.results.get("result") as RationalValue).value.equals(new Fraction(100))).toBe(true);
    expect((resultB.results.get("result") as RationalValue).value.equals(new Fraction(1))).toBe(true);
  });
});
