import type Fraction from "fraction.js";
import type { Statement } from "../analyzer/ast";

// Category enum for taxonomical ordering (lower = higher priority)
export enum Category {
  Concrete = 0,
  Pictorial = 1,
  Abstract = 2,
}

// Runtime value types
export type RationalValue = {
  kind: "rational";
  value: Fraction;
};

export type BooleanValue = {
  kind: "boolean";
  value: boolean;
};

export type ShapeValue = {
  kind: "shape";
  category: "pictorial";
  subtype: "circle" | "square";
  size: "small" | "medium" | "large";
  amount: Fraction;
};

export type FoodValue = {
  kind: "food";
  category: "concrete";
  subtype: "grape" | "pear" | "apple" | "burger";
  color: "purple" | "green" | "red" | "orange";
  amount: Fraction;
};

export type AbstractValue = {
  kind: "abstract";
  category: "abstract";
  objectType: "rational";
  value: Fraction;
};

export type CPAObject = ShapeValue | FoodValue | AbstractValue;

export type ArrayValue = {
  kind: "array";
  elements: RuntimeValue[];
};

export type OtherValue = {
  kind: "other";
  value: string;
};

export type RuntimeValue =
  | RationalValue
  | BooleanValue
  | CPAObject
  | ArrayValue
  | OtherValue;

// Node evaluation state
export type EvaluationState = "pending" | "evaluating" | "completed";

// Graph node for execution
export interface ExecutionNode {
  id: string;
  statement: Statement;
  dependencies: string[];
  dependents: string[]; // Reverse dependencies: who depends on me
  state: EvaluationState;
  result?: RuntimeValue;
}

// Type guards
export function isRational(val: RuntimeValue): val is RationalValue {
  return val.kind === "rational";
}

export function isArray(val: RuntimeValue): val is ArrayValue {
  return val.kind === "array";
}

export function isCPAObject(val: RuntimeValue): val is CPAObject {
  return val.kind === "shape" || val.kind === "food" || val.kind === "abstract";
}

export function isShape(val: RuntimeValue): val is ShapeValue {
  return val.kind === "shape";
}

export function isFood(val: RuntimeValue): val is FoodValue {
  return val.kind === "food";
}

export function isAbstract(val: RuntimeValue): val is AbstractValue {
  return val.kind === "abstract";
}

export function isOther(val: RuntimeValue): val is OtherValue {
  return val.kind === "other";
}

export function isBoolean(val: RuntimeValue): val is BooleanValue {
  return val.kind === "boolean";
}

// Get the comparable quantity from a CPA object
export function getQuantity(val: CPAObject): Fraction {
  if (val.kind === "abstract") {
    return val.value;
  }
  return val.amount;
}

// Get a unique key for CPA aggregation (category + type + subtype)
export function getCPAKey(val: CPAObject): string {
  if (val.kind === "abstract") {
    return `abstract:rational`;
  }
  if (val.kind === "shape") {
    return `pictorial:shape:${val.subtype}:${val.size}`;
  }
  if (val.kind === "food") {
    return `concrete:food:${val.subtype}:${val.color}`;
  }
  return "unknown";
}

// Get category enum value from a runtime value
export function getCategoryOrder(val: RuntimeValue): Category {
  if (val.kind === "food") return Category.Concrete;
  if (val.kind === "shape") return Category.Pictorial;
  if (val.kind === "abstract" || val.kind === "rational") return Category.Abstract;
  return Category.Abstract;
}

// Get type key for sorting
export function getTypeKey(val: RuntimeValue): string {
  if (val.kind === "food") return `food:${val.subtype}`;
  if (val.kind === "shape") return `shape:${val.subtype}`;
  if (val.kind === "abstract") return "rational";
  if (val.kind === "rational") return "rational";
  return "other";
}
