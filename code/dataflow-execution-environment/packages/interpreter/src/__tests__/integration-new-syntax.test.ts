import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { RationalValue, ArrayValue, CPAObject } from "../runtime/types";

describe("Integration (new JSON-like syntax)", () => {
  describe("Simple Addition", () => {
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

  describe("CPA Objects with new syntax", () => {
    test("parses and evaluates a simple CPA object", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source manzana = {
          "category": "concreto",
          "type": "comida",
          "subtype": "manzana",
          "quantity": 5,
          "color": "rojo"
        };
        sink resultado = manzana;
      `);

      expect(result.errors).toHaveLength(0);
      const obj = result.results.get("resultado") as CPAObject;
      expect(obj.kind).toBe("cpa");
      expect(obj.category).toBe("concreto");
      expect(obj.type).toBe("comida");
      expect(obj.subtype).toBe("manzana");
      expect(obj.quantity.equals(new Fraction(5))).toBe(true);
      expect(obj.attributes.color).toBe("rojo");
    });

    test("parses a pictorial shape", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source circulo = {
          "category": "pictorico",
          "type": "forma",
          "subtype": "circulo",
          "quantity": 3,
          "size": "grande"
        };
        sink resultado = circulo;
      `);

      expect(result.errors).toHaveLength(0);
      const obj = result.results.get("resultado") as CPAObject;
      expect(obj.kind).toBe("cpa");
      expect(obj.category).toBe("pictorico");
      expect(obj.type).toBe("forma");
      expect(obj.subtype).toBe("circulo");
      expect(obj.quantity.equals(new Fraction(3))).toBe(true);
      expect(obj.attributes.size).toBe("grande");
    });

    test("parses an abstract rational", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source numero = {
          "category": "abstracto",
          "type": "numero",
          "subtype": "racional",
          "quantity": 0.5
        };
        sink resultado = numero;
      `);

      expect(result.errors).toHaveLength(0);
      const obj = result.results.get("resultado") as CPAObject;
      expect(obj.kind).toBe("cpa");
      expect(obj.category).toBe("abstracto");
      expect(obj.type).toBe("numero");
      expect(obj.subtype).toBe("racional");
      expect(obj.quantity.equals(new Fraction(0.5))).toBe(true);
    });

    test("sum aggregates CPA objects by key", async () => {
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

      // Find the uva (should have quantity 8)
      const uva = sinkResult.elements.find(e => (e as CPAObject).subtype === "uva") as CPAObject;
      expect(uva.quantity.equals(new Fraction(8))).toBe(true);

      // Find the manzana (should have quantity 2)
      const manzana = sinkResult.elements.find(e => (e as CPAObject).subtype === "manzana") as CPAObject;
      expect(manzana.quantity.equals(new Fraction(2))).toBe(true);
    });

    test("dynamic type - vehicle example from spec", async () => {
      const interpreter = new Interpreter();
      const result = await interpreter.execute(`
        source sedan = {
          "category": "concreto",
          "type": "vehiculo",
          "subtype": "carro",
          "quantity": 2,
          "puertas": "4"
        };
        source coupe = {
          "category": "concreto",
          "type": "vehiculo",
          "subtype": "carro",
          "quantity": 1,
          "puertas": "2"
        };
        transform total_carros = sum(sedan, coupe);
        sink output_cars = total_carros;
      `);

      expect(result.errors).toHaveLength(0);
      // Since they have different attributes (puertas), they should not aggregate
      const sinkResult = result.results.get("output_cars") as ArrayValue;
      expect(sinkResult.kind).toBe("arreglo");
      expect(sinkResult.elements).toHaveLength(2);
    });
  });
});
