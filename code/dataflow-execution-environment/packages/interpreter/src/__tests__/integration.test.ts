import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { execute } from "../index";
import type { RationalValue, ArrayValue, ShapeValue, FoodValue } from "../runtime/types";

describe("Integration", () => {
  describe("Example 1: Simple Addition", () => {
    test("adds two numbers", async () => {
      const result = await execute(`
        source a = 3;
        source b = 2;
        transform add = sum(a, b);
        sink result = add;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as RationalValue;
      expect(sinkResult.kind).toBe("rational");
      expect(sinkResult.value.equals(new Fraction(5))).toBe(true);
    });
  });

  describe("Example 3: Complex Expression", () => {
    test("computes (3 + 2) * (10 - 6) = 20", async () => {
      const result = await execute(`
        source a = 3;
        source b = 2;
        source c = 10;
        source d = 6;
        transform add = sum(a, b);
        transform difference = substract(c, d);
        transform product = multiply(add, difference);
        sink result = product;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as RationalValue;
      expect(sinkResult.kind).toBe("rational");
      expect(sinkResult.value.equals(new Fraction(20))).toBe(true);
    });
  });

  describe("CPA aggregation", () => {
    test("sum aggregates matching CPA objects", async () => {
      const result = await execute(`
        source items = [
          {type: grape, color: purple, amount: 5},
          {type: grape, color: purple, amount: 3},
          {type: apple, color: red, amount: 2}
        ];
        transform total = sum(items);
        sink result = total;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("array");
      // Should have 2 elements: grape(8) and apple(2)
      expect(sinkResult.elements).toHaveLength(2);
    });

    test("multiply aggregates matching CPA objects", async () => {
      const result = await execute(`
        source items = [
          {type: circle, size: large, amount: 2},
          {type: circle, size: large, amount: 3},
          {type: square, size: small, amount: 4}
        ];
        transform product = multiply(items);
        sink result = product;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("array");
      // Should have 2 elements: circle(6) and square(4)
      expect(sinkResult.elements).toHaveLength(2);

      const circle = sinkResult.elements.find(e => (e as ShapeValue).subtype === "circle") as ShapeValue;
      const square = sinkResult.elements.find(e => (e as ShapeValue).subtype === "square") as ShapeValue;
      expect(circle.amount.equals(new Fraction(6))).toBe(true);
      expect(square.amount.equals(new Fraction(4))).toBe(true);
    });

    test("substract operates on CPA object amounts", async () => {
      const result = await execute(`
        source a = {type: grape, color: purple, amount: 10};
        source b = {type: apple, color: red, amount: 3};
        transform diff = substract(a, b);
        sink result = diff;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as FoodValue;
      expect(sinkResult.kind).toBe("food");
      expect(sinkResult.amount.equals(new Fraction(7))).toBe(true);
    });

    test("divide operates on CPA object amounts", async () => {
      const result = await execute(`
        source a = {type: circle, size: medium, amount: 12};
        source b = 4;
        transform quotient = divide(a, b);
        sink result = quotient;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ShapeValue;
      expect(sinkResult.kind).toBe("shape");
      expect(sinkResult.amount.equals(new Fraction(3))).toBe(true);
    });

    test("multiply scales CPA objects by rational factor", async () => {
      const result = await execute(`
        source myShape = {type: square, size: small, amount: 5};
        source factor = 3;
        transform scaled = multiply(myShape, factor);
        sink result = scaled;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ShapeValue;
      expect(sinkResult.kind).toBe("shape");
      expect(sinkResult.subtype).toBe("square");
      expect(sinkResult.amount.equals(new Fraction(15))).toBe(true);
    });
  });
});
