import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { CPAObject } from "../runtime/types";

describe("Arithmetic operations (integration)", () => {
  test("sum is variadic", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      source c = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      source d = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 4};
      transform total = sum(a, b, c, d);
      sink result = total;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as CPAObject;
    expect(sinkResult.quantity.equals(new Fraction(10))).toBe(true);
  });

  test("multiply is variadic", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      source c = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 4};
      transform product = multiply(a, b, c);
      sink result = product;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as CPAObject;
    expect(sinkResult.quantity.equals(new Fraction(24))).toBe(true);
  });

  test("substract is binary", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      transform diff = substract(a, b);
      sink result = diff;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as CPAObject;
    expect(sinkResult.quantity.equals(new Fraction(7))).toBe(true);
  });

  test("divide is binary", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 15};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      transform quotient = divide(a, b);
      sink result = quotient;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as CPAObject;
    expect(sinkResult.quantity.equals(new Fraction(5))).toBe(true);
  });

  test("rational arithmetic preserves precision", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      transform third = divide(a, b);
      source c = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      transform whole = multiply(third, c);
      sink result = whole;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as CPAObject;
    // 1/3 * 3 = 1 exactly with fractions
    expect(sinkResult.quantity.equals(new Fraction(1))).toBe(true);
  });
});
