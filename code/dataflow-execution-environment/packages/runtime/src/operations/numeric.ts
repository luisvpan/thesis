import type { Natural, Integer, Decimal, Boolean } from "@dataflow/shared/types";

function unwrapValue(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "object" && value !== null && "value" in value) {
    return (value as { value: number }).value;
  }
  throw new TypeError("Expected number value");
}

export function ADD(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;
  const aVal = unwrapValue(a.value);
  const bVal = unwrapValue(b.value);
  
  if (aVal < 0 || bVal < 0) {
    throw new RangeError("ADD requires natural numbers (>= 0)");
  }
  
  return {
    kind: "natural",
    value: aVal + bVal
  };
}

export function SUBTRACT(inputs: Array<{ id: string; value: unknown }>): Integer {
  const [a, b] = inputs;
  const aVal = unwrapValue(a.value);
  const bVal = unwrapValue(b.value);
  
  return {
    kind: "integer",
    value: aVal - bVal
  };
}

export function MULTIPLY(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;
  const aVal = unwrapValue(a.value);
  const bVal = unwrapValue(b.value);
  
  if (aVal < 0 || bVal < 0) {
    throw new RangeError("MULTIPLY requires natural numbers (>= 0)");
  }
  
  return {
    kind: "natural",
    value: aVal * bVal
  };
}

export function DIVIDE(inputs: Array<{ id: string; value: unknown }>): Decimal {
  const [a, b] = inputs;
  const aVal = unwrapValue(a.value);
  const bVal = unwrapValue(b.value);
  
  if (bVal === 0) {
    throw new Error("DIVIDE: Division by zero");
  }
  
  return {
    kind: "decimal",
    value: aVal / bVal
  };
}

export function COMPARE(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [a, b] = inputs;
  const aVal = unwrapValue(a.value);
  const bVal = unwrapValue(b.value);
  
  return {
    kind: "boolean",
    value: aVal === bVal
  };
}
