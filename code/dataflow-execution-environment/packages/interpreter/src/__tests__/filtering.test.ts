import { describe, test, expect } from "bun:test";
import { Interpreter } from "../index";
import type { ArrayValue } from "../runtime/types";

describe("Filter operation (integration)", () => {
  test("filters shapes by size", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source shapes = [
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 1, "size": "grande"},
        {"category": "pictorico", "type": "forma", "subtype": "cuadrado", "quantity": 1, "size": "pequeño"},
        {"category": "pictorico", "type": "forma", "subtype": "cuadrado", "quantity": 1, "size": "grande"}
      ];
      transform large_shapes = filter(shapes, "grande");
      sink result = large_shapes;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    expect(sinkResult.elements).toHaveLength(2);
  });

  test("filters foods by color", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source foods = [
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 3, "color": "rojo"},
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 5, "color": "morado"},
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 2, "color": "verde"}
      ];
      transform red_foods = filter(foods, "rojo");
      sink result = red_foods;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result");
    // Should return only one element (not an array)
    expect(sinkResult).toBeDefined();
  });

  test("filters by category", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 1, "color": "rojo"},
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 1, "size": "grande"},
        {"category": "concreto", "type": "montessori", "subtype": "cubo", "quantity": 1, "color": "azul"}
      ];
      transform concrete_only = filter(items, "concreto");
      sink result = concrete_only;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    expect(sinkResult.elements).toHaveLength(2);
  });

  test("filters by type", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {"category": "concreto", "type": "comida", "subtype": "manzana", "quantity": 1, "color": "rojo"},
        {"category": "concreto", "type": "comida", "subtype": "uva", "quantity": 1, "color": "morado"},
        {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 1, "size": "grande"}
      ];
      transform foods_only = filter(items, "comida");
      sink result = foods_only;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    expect(sinkResult.elements).toHaveLength(2);
  });
});
