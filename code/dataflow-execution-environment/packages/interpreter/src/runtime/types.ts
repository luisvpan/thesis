import type Fraction from "fraction.js";
import type { Statement, ShapeTypeValue, SizeValue, FoodTypeValue, ColorValue } from "../analyzer/ast";

// Category enum for taxonomical ordering (lower = higher priority)
export enum Category {
  Concreto = 0,
  Pictorico = 1,
  Abstracto = 2,
}

// Runtime value types
export type RationalValue = {
  kind: "racional";
  value: Fraction;
};

export type BooleanValue = {
  kind: "booleano";
  value: boolean;
};

export type ShapeValue = {
  kind: "forma";
  category: "pictorico";
  subtype: ShapeTypeValue;
  size: SizeValue;
  amount: Fraction;
};

export type FoodValue = {
  kind: "comida";
  category: "concreto";
  subtype: FoodTypeValue;
  color: ColorValue;
  amount: Fraction;
};

export type MontessoriValue = {
  kind: "montessori";
  category: "concreto";
  color: ColorValue;
  amount: Fraction;
};

export type CapValue = {
  kind: "cap";
  category: "concreto";
  color: ColorValue;
  amount: Fraction;
};

export type StickValue = {
  kind: "stick";
  category: "concreto";
  color: ColorValue;
  amount: Fraction;
};

export type AbstractValue = {
  kind: "abstracto";
  category: "abstracto";
  objectType: "racional";
  value: Fraction;
};

export type CPAObject = ShapeValue | FoodValue | MontessoriValue | CapValue | StickValue | AbstractValue;

export type ArrayValue = {
  kind: "arreglo";
  elements: RuntimeValue[];
};

export type OtherValue = {
  kind: "otro";
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
  return val.kind === "racional";
}

export function isArray(val: RuntimeValue): val is ArrayValue {
  return val.kind === "arreglo";
}

export function isCPAObject(val: RuntimeValue): val is CPAObject {
  return val.kind === "forma" || val.kind === "comida" || val.kind === "montessori" || val.kind === "cap" || val.kind === "stick" || val.kind === "abstracto";
}

export function isShape(val: RuntimeValue): val is ShapeValue {
  return val.kind === "forma";
}

export function isFood(val: RuntimeValue): val is FoodValue {
  return val.kind === "comida";
}

export function isMontessori(val: RuntimeValue): val is MontessoriValue {
  return val.kind === "montessori";
}

export function isCap(val: RuntimeValue): val is CapValue {
  return val.kind === "cap";
}

export function isStick(val: RuntimeValue): val is StickValue {
  return val.kind === "stick";
}

export function isAbstract(val: RuntimeValue): val is AbstractValue {
  return val.kind === "abstracto";
}

export function isOther(val: RuntimeValue): val is OtherValue {
  return val.kind === "otro";
}

export function isBoolean(val: RuntimeValue): val is BooleanValue {
  return val.kind === "booleano";
}

// Get a unique key for CPA aggregation (category + type + subtype)
export function getCPAKey(val: CPAObject): string {
  if (val.kind === "abstracto") {
    return `abstracto:racional`;
  }
  if (val.kind === "forma") {
    return `pictorico:forma:${val.subtype}:${val.size}`;
  }
  if (val.kind === "comida") {
    return `concreto:comida:${val.subtype}:${val.color}`;
  }
  if (val.kind === "montessori") {
    return `concreto:montessori:${val.color}`;
  }
  if (val.kind === "cap") {
    return `concreto:cap:${val.color}`;
  }
  if (val.kind === "stick") {
    return `concreto:stick:${val.color}`;
  }
  return "unknown";
}

// Get category enum value from a runtime value
export function getCategoryOrder(val: RuntimeValue): Category {
  if (val.kind === "comida" || val.kind === "montessori" || val.kind === "cap" || val.kind === "stick") return Category.Concreto;
  if (val.kind === "forma") return Category.Pictorico;
  if (val.kind === "abstracto" || val.kind === "racional") return Category.Abstracto;
  return Category.Abstracto;
}

// Get type key for sorting
export function getTypeKey(val: RuntimeValue): string {
  if (val.kind === "comida") return `comida:${val.subtype}`;
  if (val.kind === "forma") return `forma:${val.subtype}`;
  if (val.kind === "montessori") return `montessori:${val.color}`;
  if (val.kind === "cap") return `cap:${val.color}`;
  if (val.kind === "stick") return `stick:${val.color}`;
  if (val.kind === "abstracto") return "racional";
  if (val.kind === "racional") return "racional";
  return "otro";
}
