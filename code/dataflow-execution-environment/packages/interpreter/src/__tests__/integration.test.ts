import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { RationalValue, ArrayValue, CPAObject } from "../runtime/types";

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
          {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 5, "color": "morado"},
          {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 3, "color": "morado"},
          {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 2, "color": "rojo"}
        ];
        transform total = sum(items);
        sink result = total;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(2);
    });

    test("multiply does not aggregate CPA objects, returns all as array", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source items = [
          {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 2, "size": "grande"},
          {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 3, "size": "grande"},
          {"category": "pictorico", "type": "forma", "subtype": "cuadrado", "quantity": 4, "size": "pequeño"}
        ];
        transform product = multiply(items);
        sink result = product;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      // CPAs are NOT combined - all 3 elements are preserved
      expect(sinkResult.elements).toHaveLength(3);

      const circles = sinkResult.elements.filter(e => (e as CPAObject).subtype === "circulo") as CPAObject[];
      const square = sinkResult.elements.find(e => (e as CPAObject).subtype === "cuadrado") as CPAObject;
      expect(circles).toHaveLength(2);
      expect(circles[0].quantity.equals(new Fraction(2))).toBe(true);
      expect(circles[1].quantity.equals(new Fraction(3))).toBe(true);
      expect(square.quantity.equals(new Fraction(4))).toBe(true);
    });

    test("substract operates on CPA object amounts", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source a = {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 10, "color": "morado"};
        source b = {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 3, "color": "rojo"};
        transform diff = substract(a, b);
        sink result = diff;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as CPAObject;
      expect(sinkResult.kind).toBe("cpa");
      expect(sinkResult.quantity.equals(new Fraction(7))).toBe(true);
    });

    test("divide operates on CPA object amounts", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source a = {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 12, "size": "mediano"};
        source b = 4;
        transform quotient = divide(a, b);
        sink result = quotient;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as CPAObject;
      expect(sinkResult.kind).toBe("cpa");
      expect(sinkResult.quantity.equals(new Fraction(3))).toBe(true);
    });

    test("multiply scales CPA objects by rational factor", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source myShape = {"category": "pictorico", "type": "forma", "subtype": "cuadrado", "quantity": 5, "size": "pequeño"};
        source factor = 3;
        transform scaled = multiply(myShape, factor);
        sink result = scaled;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("result") as CPAObject;
      expect(sinkResult.kind).toBe("cpa");
      expect(sinkResult.subtype).toBe("cuadrado");
      expect(sinkResult.quantity.equals(new Fraction(15))).toBe(true);
    });
  });

  describe("Montessori cubes", () => {
    test("sum aggregates montessori cubes by color", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source rojos = {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 3, "color": "rojo"};
        source azules = {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 5, "color": "azul"};
        transform total = sum(rojos, azules);
        sink resultado = total;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("resultado") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(2);

      const rojos = sinkResult.elements.find(e => (e as CPAObject).attributes.color === "rojo") as CPAObject;
      const azules = sinkResult.elements.find(e => (e as CPAObject).attributes.color === "azul") as CPAObject;
      expect(rojos.kind).toBe("cpa");
      expect(rojos.quantity.equals(new Fraction(3))).toBe(true);
      expect(azules.kind).toBe("cpa");
      expect(azules.quantity.equals(new Fraction(5))).toBe(true);
    });

    test("sum merges array source built from identifiers with another montessori", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source orange = {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 1, "color": "naranja"};
        source purple = {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 1, "color": "morado"};
        source red = {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 1, "color": "rojo"};
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
        source cubo = {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 1, "color": "verde"};
        sink resultado = cubo;
      `);

      expect(result.errors).toHaveLength(0);
      const sinkResult = result.results.get("resultado") as CPAObject;
      expect(sinkResult.kind).toBe("cpa");
      expect(sinkResult.category).toBe("concreto");
      expect(sinkResult.attributes.color).toBe("verde");
    });
  });
});
