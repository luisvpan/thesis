import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { filter } from "../filtering";
import type { ShapeValue, FoodValue, ArrayValue, OtherValue } from "../../runtime/types";

describe("filter (unit)", () => {
  test("filters shapes by size", () => {
    const shapes: ShapeValue[] = [
      { kind: "shape", category: "pictorial", subtype: "circle", size: "large", amount: new Fraction(1) },
      { kind: "shape", category: "pictorial", subtype: "square", size: "small", amount: new Fraction(1) },
      { kind: "shape", category: "pictorial", subtype: "square", size: "large", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "other", value: "large" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("array");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as ShapeValue).subtype).toBe("circle");
    expect((result.elements[1] as ShapeValue).subtype).toBe("square");
  });

  test("filters shapes by subtype", () => {
    const shapes: ShapeValue[] = [
      { kind: "shape", category: "pictorial", subtype: "circle", size: "large", amount: new Fraction(1) },
      { kind: "shape", category: "pictorial", subtype: "square", size: "small", amount: new Fraction(1) },
      { kind: "shape", category: "pictorial", subtype: "circle", size: "medium", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "other", value: "circle" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("array");
    expect(result.elements).toHaveLength(2);
  });

  test("filters foods by color", () => {
    const foods: FoodValue[] = [
      { kind: "food", category: "concrete", subtype: "grape", color: "purple", amount: new Fraction(5) },
      { kind: "food", category: "concrete", subtype: "apple", color: "red", amount: new Fraction(3) },
      { kind: "food", category: "concrete", subtype: "grape", color: "purple", amount: new Fraction(2) },
    ];
    const criterion: OtherValue = { kind: "other", value: "purple" };
    const result = filter([...foods, criterion]) as ArrayValue;
    expect(result.kind).toBe("array");
    expect(result.elements).toHaveLength(2);
  });

  test("filters foods by subtype", () => {
    const foods: FoodValue[] = [
      { kind: "food", category: "concrete", subtype: "grape", color: "purple", amount: new Fraction(5) },
      { kind: "food", category: "concrete", subtype: "apple", color: "red", amount: new Fraction(3) },
      { kind: "food", category: "concrete", subtype: "pear", color: "green", amount: new Fraction(2) },
    ];
    const criterion: OtherValue = { kind: "other", value: "apple" };
    const result = filter([...foods, criterion]) as FoodValue;
    expect(result.kind).toBe("food");
    expect(result.subtype).toBe("apple");
  });

  test("returns single element when only one matches", () => {
    const shapes: ShapeValue[] = [
      { kind: "shape", category: "pictorial", subtype: "circle", size: "large", amount: new Fraction(1) },
      { kind: "shape", category: "pictorial", subtype: "square", size: "small", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "other", value: "small" };
    const result = filter([...shapes, criterion]) as ShapeValue;
    expect(result.kind).toBe("shape");
    expect(result.subtype).toBe("square");
  });

  test("returns empty array when nothing matches", () => {
    const shapes: ShapeValue[] = [
      { kind: "shape", category: "pictorial", subtype: "circle", size: "large", amount: new Fraction(1) },
      { kind: "shape", category: "pictorial", subtype: "square", size: "large", amount: new Fraction(1) },
    ];
    const criterion: OtherValue = { kind: "other", value: "small" };
    const result = filter([...shapes, criterion]) as ArrayValue;
    expect(result.kind).toBe("array");
    expect(result.elements).toHaveLength(0);
  });

  test("filters by category", () => {
    const grape: FoodValue = {
      kind: "food",
      category: "concrete",
      subtype: "grape",
      color: "purple",
      amount: new Fraction(5),
    };
    const circle: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "large",
      amount: new Fraction(3),
    };
    const criterion: OtherValue = { kind: "other", value: "concrete" };
    const result = filter([grape, circle, criterion]) as FoodValue;
    expect(result.kind).toBe("food");
    expect(result.category).toBe("concrete");
  });

  test("throws error for insufficient arguments", () => {
    const shape: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "large",
      amount: new Fraction(1),
    };
    expect(() => filter([shape])).toThrow();
  });

  test("throws error for non-keyword criterion", () => {
    const shape: ShapeValue = {
      kind: "shape",
      category: "pictorial",
      subtype: "circle",
      size: "large",
      amount: new Fraction(1),
    };
    const badCriterion = { kind: "rational", value: new Fraction(5) };
    expect(() => filter([shape, badCriterion as any])).toThrow();
  });
});
