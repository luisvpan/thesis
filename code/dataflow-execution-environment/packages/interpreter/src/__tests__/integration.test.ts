import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { RationalValue, ArrayValue, ShapeValue, FoodValue, MontessoriValue } from "../runtime/types";

describe("Integration", () => {
  describe("Example 1: Simple Addition", () => {
    test("adds two numbers", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source a = 3;
        source b = 2;
        transform add = sum(a, b);
        sink result = add;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as RationalValue;
      expect(sinkResult.kind).toBe("racional");
      expect(sinkResult.value.equals(new Fraction(5))).toBe(true);
    });
  });

  describe("Example 3: Complex Expression", () => {
    test("computes (3 + 2) * (10 - 6) = 20", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
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
      expect(sinkResult.kind).toBe("racional");
      expect(sinkResult.value.equals(new Fraction(20))).toBe(true);
    });
  });

  describe("CPA aggregation", () => {
    test("sum aggregates matching CPA objects", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source items = [
          {type: uva, color: morado, amount: 5},
          {type: uva, color: morado, amount: 3},
          {type: manzana, color: rojo, amount: 2}
        ];
        transform total = sum(items);
        sink result = total;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(2);
    });

    test("multiply aggregates matching CPA objects", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source items = [
          {type: circulo, size: grande, amount: 2},
          {type: circulo, size: grande, amount: 3},
          {type: cuadrado, size: pequeño, amount: 4}
        ];
        transform product = multiply(items);
        sink result = product;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(2);

      const circle = sinkResult.elements.find(e => (e as ShapeValue).subtype === "circulo") as ShapeValue;
      const square = sinkResult.elements.find(e => (e as ShapeValue).subtype === "cuadrado") as ShapeValue;
      expect(circle.amount.equals(new Fraction(6))).toBe(true);
      expect(square.amount.equals(new Fraction(4))).toBe(true);
    });

    test("substract operates on CPA object amounts", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source a = {type: uva, color: morado, amount: 10};
        source b = {type: manzana, color: rojo, amount: 3};
        transform diff = substract(a, b);
        sink result = diff;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as FoodValue;
      expect(sinkResult.kind).toBe("comida");
      expect(sinkResult.amount.equals(new Fraction(7))).toBe(true);
    });

    test("divide operates on CPA object amounts", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source a = {type: circulo, size: mediano, amount: 12};
        source b = 4;
        transform quotient = divide(a, b);
        sink result = quotient;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ShapeValue;
      expect(sinkResult.kind).toBe("forma");
      expect(sinkResult.amount.equals(new Fraction(3))).toBe(true);
    });

    test("multiply scales CPA objects by rational factor", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source myShape = {type: cuadrado, size: pequeño, amount: 5};
        source factor = 3;
        transform scaled = multiply(myShape, factor);
        sink result = scaled;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ShapeValue;
      expect(sinkResult.kind).toBe("forma");
      expect(sinkResult.subtype).toBe("cuadrado");
      expect(sinkResult.amount.equals(new Fraction(15))).toBe(true);
    });
  });

  describe("Montessori cubes", () => {
    test("sum aggregates montessori cubes by color", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source rojos = { type: montessori, color: rojo, amount: 3 };
        source azules = { type: montessori, color: azul, amount: 5 };
        transform total = sum(rojos, azules);
        sink resultado = total;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("resultado") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(2);

      const rojos = sinkResult.elements.find(e => (e as MontessoriValue).color === "rojo") as MontessoriValue;
      const azules = sinkResult.elements.find(e => (e as MontessoriValue).color === "azul") as MontessoriValue;
      expect(rojos.kind).toBe("montessori");
      expect(rojos.amount.equals(new Fraction(3))).toBe(true);
      expect(azules.kind).toBe("montessori");
      expect(azules.amount.equals(new Fraction(5))).toBe(true);
    });

    test("sum merges array source built from identifiers with another montessori", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source orange = { type: montessori, color: naranja, amount: 1 };
        source purple = { type: montessori, color: morado, amount: 1 };
        source red = { type: montessori, color: rojo, amount: 1 };
        source duo = [orange, purple];
        transform added = sum(duo, red);
        sink result = added;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(3);
    });

    test("montessori cubes are concrete category", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source cubo = { type: montessori, color: verde, amount: 1 };
        sink resultado = cubo;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("resultado") as MontessoriValue;
      expect(sinkResult.kind).toBe("montessori");
      expect(sinkResult.category).toBe("concreto");
      expect(sinkResult.color).toBe("verde");
    });
  });
});
