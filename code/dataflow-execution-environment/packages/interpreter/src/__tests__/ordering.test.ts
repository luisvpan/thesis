import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { Interpreter } from "../index";
import type { ArrayValue, ShapeValue, FoodValue } from "../runtime/types";

describe("Ordering operations (integration)", () => {
  test("order_asc sorts by taxonomical rules", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {type: uva, color: morado, amount: 5},
        {type: circulo, size: grande, amount: 3},
        {type: manzana, color: rojo, amount: 1}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    // Concrete (food) < Pictorial (shape)
    expect((sinkResult.elements[0] as FoodValue).kind).toBe("comida");
    expect((sinkResult.elements[1] as FoodValue).kind).toBe("comida");
    expect((sinkResult.elements[2] as ShapeValue).kind).toBe("forma");
  });

  test("order_desc sorts by taxonomical rules in reverse", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {type: uva, color: morado, amount: 5},
        {type: circulo, size: grande, amount: 3},
        {type: manzana, color: rojo, amount: 1}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    // Descending: Abstract > Pictorial > Concrete
    expect((sinkResult.elements[0] as ShapeValue).kind).toBe("forma");
    expect((sinkResult.elements[1] as FoodValue).kind).toBe("comida");
    expect((sinkResult.elements[2] as FoodValue).kind).toBe("comida");
  });

  test("order_asc sorts by type alphabetically within category", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {type: pera, color: verde, amount: 1},
        {type: manzana, color: rojo, amount: 2},
        {type: uva, color: morado, amount: 3}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as FoodValue).subtype).toBe("manzana");
    expect((sinkResult.elements[1] as FoodValue).subtype).toBe("pera");
    expect((sinkResult.elements[2] as FoodValue).subtype).toBe("uva");
  });

  test("order_asc sorts by quantity within same type", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {type: uva, color: morado, amount: 10},
        {type: uva, color: morado, amount: 2},
        {type: uva, color: morado, amount: 5}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as FoodValue).amount.equals(new Fraction(2))).toBe(true);
    expect((sinkResult.elements[1] as FoodValue).amount.equals(new Fraction(5))).toBe(true);
    expect((sinkResult.elements[2] as FoodValue).amount.equals(new Fraction(10))).toBe(true);
  });

  test("order_desc sorts by quantity in reverse within same type", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {type: circulo, size: grande, amount: 2},
        {type: circulo, size: grande, amount: 10},
        {type: circulo, size: grande, amount: 5}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as ShapeValue).amount.equals(new Fraction(10))).toBe(true);
    expect((sinkResult.elements[1] as ShapeValue).amount.equals(new Fraction(5))).toBe(true);
    expect((sinkResult.elements[2] as ShapeValue).amount.equals(new Fraction(2))).toBe(true);
  });

  test("order_desc sorts by type alphabetically in reverse within category", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source items = [
        {type: manzana, color: rojo, amount: 1},
        {type: pera, color: verde, amount: 2},
        {type: uva, color: morado, amount: 3}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect((sinkResult.elements[0] as FoodValue).subtype).toBe("uva");
    expect((sinkResult.elements[1] as FoodValue).subtype).toBe("pera");
    expect((sinkResult.elements[2] as FoodValue).subtype).toBe("manzana");
  });
});
