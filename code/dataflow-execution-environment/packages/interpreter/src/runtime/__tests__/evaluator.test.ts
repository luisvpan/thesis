import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../../index";
import type { RationalValue } from "../types";

describe("Lazy evaluation", () => {
  test("memoizes node results", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source x = 5;
      transform a = sum(x, x);
      transform b = sum(a, a);
      transform c = sum(b, b);
      sink result = c;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as RationalValue;
    // 5 + 5 = 10, 10 + 10 = 20, 20 + 20 = 40
    expect(sinkResult.value.equals(new Fraction(40))).toBe(true);
  });
});

describe("Incremental re-evaluation", () => {
  test("first execution evaluates all nodes", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = 5;
      source b = 3;
      transform sum_ab = sum(a, b);
      sink result = sum_ab;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(4);
    expect(stats.evaluated).toBe(4); // All nodes evaluated first time
    expect(stats.cached).toBe(0);
  });

  test("reuses cache for unchanged nodes", async () => {
    const interpreter = new Interpreter();

    // First execution - all nodes evaluated
    await interpreter.execute(`
      source a = 5;
      source b = 3;
      transform sum_ab = sum(a, b);
      sink result = sum_ab;
    `);

    // Change only 'b', 'a' should be reused from cache
    const result2 = await interpreter.execute(`
      source a = 5;
      source b = 10;
      transform sum_ab = sum(a, b);
      sink result = sum_ab;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.cached).toBe(1);    // 'a' reused from cache
    expect(stats.evaluated).toBe(3); // b, sum_ab, result re-evaluated

    expect((result2.results.get("result") as RationalValue).value.equals(new Fraction(15))).toBe(true);
  });

  test("invalidates dependents when source changes", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source x = 2;
      transform doubled = multiply(x, 2);
      transform quadrupled = multiply(doubled, 2);
      sink result = quadrupled;
    `);

    // Change x → all dependents (doubled, quadrupled, result) must be re-evaluated
    await interpreter.execute(`
      source x = 5;
      transform doubled = multiply(x, 2);
      transform quadrupled = multiply(doubled, 2);
      sink result = quadrupled;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.evaluated).toBe(4); // All nodes depend on x
    expect(stats.cached).toBe(0);
  });

  test("handles added nodes", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = 5;
      sink result = a;
    `);

    const result = await interpreter.execute(`
      source a = 5;
      source b = 3;
      transform total = sum(a, b);
      sink result = total;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(4);
    expect(stats.cached).toBe(1);    // 'a' reused
    expect(stats.evaluated).toBe(3); // b, total, result are new/changed

    expect((result.results.get("result") as RationalValue).value.equals(new Fraction(8))).toBe(true);
  });

  test("handles removed nodes", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = 5;
      source b = 3;
      transform total = sum(a, b);
      sink result = total;
    `);

    const result = await interpreter.execute(`
      source a = 5;
      sink result = a;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(2);
    expect(stats.cached).toBe(1);    // 'a' reused
    expect(stats.evaluated).toBe(1); // result changed (now points to a)

    expect((result.results.get("result") as RationalValue).value.equals(new Fraction(5))).toBe(true);
  });

  test("interpreter.reset() clears all state", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`source x = 100; sink result = x;`);
    expect(interpreter.getCacheSize()).toBe(2);

    interpreter.reset();
    expect(interpreter.getCacheSize()).toBe(0);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(0);
    expect(stats.evaluated).toBe(0);
    expect(stats.cached).toBe(0);

    const result = await interpreter.execute(`source y = 1; sink result = y;`);
    expect((result.results.get("result") as RationalValue).value.equals(new Fraction(1))).toBe(true);
  });

  test("multiple interpreter instances are independent", async () => {
    const interpreter1 = new Interpreter();
    const interpreter2 = new Interpreter();

    await interpreter1.execute(`source x = 100; sink result = x;`);
    await interpreter2.execute(`source y = 200; sink result = y;`);

    // Re-execute with changes
    const result1 = await interpreter1.execute(`source x = 101; sink result = x;`);
    const result2 = await interpreter2.execute(`source y = 201; sink result = y;`);

    // Both should have re-evaluated all nodes (value changed)
    expect(interpreter1.getEvaluationStats().evaluated).toBe(2);
    expect(interpreter2.getEvaluationStats().evaluated).toBe(2);

    expect((result1.results.get("result") as RationalValue).value.equals(new Fraction(101))).toBe(true);
    expect((result2.results.get("result") as RationalValue).value.equals(new Fraction(201))).toBe(true);
  });
});
