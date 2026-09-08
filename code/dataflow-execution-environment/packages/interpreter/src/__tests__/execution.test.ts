import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { CPAObject } from "../runtime/types";

describe("Independent executions", () => {
  test("re-executes with modified source values", async () => {
    const interpreter1 = new Interpreter();
    const result1 = await interpreter1.execute(`
      source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};
      source two = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      transform doubled = multiply(x, two);
      sink result = doubled;
    `);
    expect((result1.results.get("result") as CPAObject).quantity.equals(new Fraction(20))).toBe(true);

    const interpreter2 = new Interpreter();
    const result2 = await interpreter2.execute(`
      source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 50};
      source two = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      transform doubled = multiply(x, two);
      sink result = doubled;
    `);
    expect((result2.results.get("result") as CPAObject).quantity.equals(new Fraction(100))).toBe(true);
  });

  test("re-executes with added statements", async () => {
    const interpreter1 = new Interpreter();
    const result1 = await interpreter1.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      sink result = a;
    `);
    expect((result1.results.get("result") as CPAObject).quantity.equals(new Fraction(5))).toBe(true);

    const interpreter2 = new Interpreter();
    const result2 = await interpreter2.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      transform total = sum(a, b);
      sink result = total;
    `);
    expect((result2.results.get("result") as CPAObject).quantity.equals(new Fraction(8))).toBe(true);
  });

  test("re-executes with removed statements", async () => {
    const interpreter1 = new Interpreter();
    const result1 = await interpreter1.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      transform total = sum(a, b);
      sink result = total;
    `);
    expect((result1.results.get("result") as CPAObject).quantity.equals(new Fraction(8))).toBe(true);

    const interpreter2 = new Interpreter();
    const result2 = await interpreter2.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      sink result = a;
    `);
    expect((result2.results.get("result") as CPAObject).quantity.equals(new Fraction(5))).toBe(true);
  });

  test("re-executes with changed operations", async () => {
    const interpreter1 = new Interpreter();
    const result1 = await interpreter1.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      transform calc = sum(a, b);
      sink result = calc;
    `);
    expect((result1.results.get("result") as CPAObject).quantity.equals(new Fraction(12))).toBe(true);

    const interpreter2 = new Interpreter();
    const result2 = await interpreter2.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      transform calc = multiply(a, b);
      sink result = calc;
    `);
    expect((result2.results.get("result") as CPAObject).quantity.equals(new Fraction(20))).toBe(true);
  });

  test("executions are independent (no state leakage)", async () => {
    const interpreterA = new Interpreter();
    const resultA = await interpreterA.execute(`
      source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 100};
      sink result = x;
    `);

    const interpreterB = new Interpreter();
    const resultB = await interpreterB.execute(`
      source y = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      sink result = y;
    `);

    expect((resultA.results.get("result") as CPAObject).quantity.equals(new Fraction(100))).toBe(true);
    expect((resultB.results.get("result") as CPAObject).quantity.equals(new Fraction(1))).toBe(true);
  });
});
