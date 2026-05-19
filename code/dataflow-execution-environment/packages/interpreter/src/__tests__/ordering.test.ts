import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { ArrayValue, CPAObject } from "../runtime/types";

describe("Ordering operations (integration)", () => {
  test("order_asc sorts by taxonomical rules", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 5, "color": "morado"},
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 3, "size": "grande"},
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 1, "color": "rojo"}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    // Concrete (food) < Pictorial (shape)
    expect((sinkResult.elements[0] as CPAObject).category).toBe("concreto");
    expect((sinkResult.elements[1] as CPAObject).category).toBe("concreto");
    expect((sinkResult.elements[2] as CPAObject).category).toBe("pictorico");
  });

  test("order_desc sorts by taxonomical rules in reverse", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 5, "color": "morado"},
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 3, "size": "grande"},
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 1, "color": "rojo"}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    // Descending: Abstract > Pictorial > Concrete
    expect((sinkResult.elements[0] as CPAObject).category).toBe("pictorico");
    expect((sinkResult.elements[1] as CPAObject).category).toBe("concreto");
    expect((sinkResult.elements[2] as CPAObject).category).toBe("concreto");
  });

  test("order_asc sorts by type alphabetically within category", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "pera", "quantity": 1, "color": "verde"},
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 2, "color": "rojo"},
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 3, "color": "morado"}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as CPAObject).subtype).toBe("manzana");
    expect((sinkResult.elements[1] as CPAObject).subtype).toBe("pera");
    expect((sinkResult.elements[2] as CPAObject).subtype).toBe("uva");
  });

  test("order_asc sorts by quantity within same type", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 10, "color": "morado"},
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 2, "color": "morado"},
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 5, "color": "morado"}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as CPAObject).quantity.equals(new Fraction(2))).toBe(true);
    expect((sinkResult.elements[1] as CPAObject).quantity.equals(new Fraction(5))).toBe(true);
    expect((sinkResult.elements[2] as CPAObject).quantity.equals(new Fraction(10))).toBe(true);
  });

  test("order_desc sorts by quantity in reverse within same type", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 2, "size": "grande"},
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 10, "size": "grande"},
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 5, "size": "grande"}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as CPAObject).quantity.equals(new Fraction(10))).toBe(true);
    expect((sinkResult.elements[1] as CPAObject).quantity.equals(new Fraction(5))).toBe(true);
    expect((sinkResult.elements[2] as CPAObject).quantity.equals(new Fraction(2))).toBe(true);
  });

  test("order_desc sorts by type alphabetically in reverse within category", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 1, "color": "rojo"},
        {"category": "concreto", "type": "comida", "subtype": "pera", "quantity": 2, "color": "verde"},
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 3, "color": "morado"}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as CPAObject).subtype).toBe("uva");
    expect((sinkResult.elements[1] as CPAObject).subtype).toBe("pera");
    expect((sinkResult.elements[2] as CPAObject).subtype).toBe("manzana");
  });
});
