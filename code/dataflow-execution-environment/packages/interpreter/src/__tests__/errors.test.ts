import { describe, test, expect } from "bun:test";
import { Interpreter } from "../index";

describe("Error handling", () => {
  test("detects circular dependencies", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      transform a = sum(b, x);
      transform b = sum(a, x);
      sink result = a;
    `);

    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("detects undefined references", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source one = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      transform a = sum(undefined_var, one);
      sink result = a;
    `);

    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("division by zero throws error", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 0};
      transform bad = divide(a, b);
      sink result = bad;
    `);

    expect(result.errors.length).toBeGreaterThan(0);
  });
});
