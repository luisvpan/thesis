import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { execute } from "../../index";
import type { ArrayValue, ShapeValue, FoodValue } from "../../runtime/types";

describe("Ordering operations", () => {
  test("order_asc sorts by taxonomical rules", async () => {
    const result = await execute(`
      source items = [
        {type: grape, color: purple, amount: 5},
        {type: circle, size: large, amount: 3},
        {type: apple, color: red, amount: 1}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("array");
    // Concrete (food) < Pictorial (shape)
    // So foods come first, then shapes
    expect((sinkResult.elements[0] as FoodValue).kind).toBe("food");
    expect((sinkResult.elements[1] as FoodValue).kind).toBe("food");
    expect((sinkResult.elements[2] as ShapeValue).kind).toBe("shape");
  });

  test("order_desc sorts by taxonomical rules in reverse", async () => {
    const result = await execute(`
      source items = [
        {type: grape, color: purple, amount: 5},
        {type: circle, size: large, amount: 3},
        {type: apple, color: red, amount: 1}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("array");
    // Descending: Abstract > Pictorial > Concrete
    // So shapes come first, then foods
    expect((sinkResult.elements[0] as ShapeValue).kind).toBe("shape");
    expect((sinkResult.elements[1] as FoodValue).kind).toBe("food");
    expect((sinkResult.elements[2] as FoodValue).kind).toBe("food");
  });

  test("order_asc sorts by type alphabetically within category", async () => {
    const result = await execute(`
      source items = [
        {type: pear, color: green, amount: 1},
        {type: apple, color: red, amount: 2},
        {type: grape, color: purple, amount: 3}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    // All concrete/food, sorted by subtype alphabetically
    expect((sinkResult.elements[0] as FoodValue).subtype).toBe("apple");
    expect((sinkResult.elements[1] as FoodValue).subtype).toBe("grape");
    expect((sinkResult.elements[2] as FoodValue).subtype).toBe("pear");
  });

  test("order_asc sorts by quantity within same type", async () => {
    const result = await execute(`
      source items = [
        {type: grape, color: purple, amount: 10},
        {type: grape, color: purple, amount: 2},
        {type: grape, color: purple, amount: 5}
      ];
      transform sorted = order_asc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    // Same type, sorted by amount
    expect((sinkResult.elements[0] as FoodValue).amount.equals(new Fraction(2))).toBe(true);
    expect((sinkResult.elements[1] as FoodValue).amount.equals(new Fraction(5))).toBe(true);
    expect((sinkResult.elements[2] as FoodValue).amount.equals(new Fraction(10))).toBe(true);
  });

  test("order_desc sorts by quantity in reverse within same type", async () => {
    const result = await execute(`
      source items = [
        {type: circle, size: large, amount: 2},
        {type: circle, size: large, amount: 10},
        {type: circle, size: large, amount: 5}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    // Same type, sorted by amount descending
    expect((sinkResult.elements[0] as ShapeValue).amount.equals(new Fraction(10))).toBe(true);
    expect((sinkResult.elements[1] as ShapeValue).amount.equals(new Fraction(5))).toBe(true);
    expect((sinkResult.elements[2] as ShapeValue).amount.equals(new Fraction(2))).toBe(true);
  });

  test("order_desc sorts by type alphabetically in reverse within category", async () => {
    const result = await execute(`
      source items = [
        {type: apple, color: red, amount: 1},
        {type: pear, color: green, amount: 2},
        {type: grape, color: purple, amount: 3}
      ];
      transform sorted = order_desc(items);
      sink result = sorted;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    // All concrete/food, sorted by subtype reverse alphabetically: pear > grape > apple
    expect((sinkResult.elements[0] as FoodValue).subtype).toBe("pear");
    expect((sinkResult.elements[1] as FoodValue).subtype).toBe("grape");
    expect((sinkResult.elements[2] as FoodValue).subtype).toBe("apple");
  });
});
