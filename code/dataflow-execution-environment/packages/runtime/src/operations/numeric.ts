import type { Natural } from "@dataflow/shared/types";

export function ADD(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;

  if (typeof a.value !== "number" || typeof b.value !== "number") {
    throw new TypeError("ADD requires number inputs");
  }

  if (a.value < 0 || b.value < 0) {
    throw new RangeError("ADD requires natural numbers (>= 0)");
  }

  return {
    kind: "natural",
    value: a.value + b.value
  };
}

export function SUBTRACT(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;

  if (typeof a.value !== "number" || typeof b.value !== "number") {
    throw new TypeError("SUBTRACT requires number inputs");
  }

  return {
    kind: "natural",
    value: a.value - b.value
  };
}

export function MULTIPLY(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;

  if (typeof a.value !== "number" || typeof b.value !== "number") {
    throw new TypeError("MULTIPLY requires number inputs");
  }

  if (a.value < 0 || b.value < 0) {
    throw new RangeError("MULTIPLY requires natural numbers (>= 0)");
  }

  return {
    kind: "natural",
    value: a.value * b.value
  };
}

export function DIVIDE(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;

  if (typeof a.value !== "number" || typeof b.value !== "number") {
    throw new TypeError("DIVIDE requires number inputs");
  }

  if (b.value === 0) {
    throw new Error("DIVIDE: Division by zero");
  }

  return {
    kind: "natural",
    value: a.value / b.value
  };
}

export function COMPARE(inputs: Array<{ id: string; value: unknown }>): Natural {
  const [a, b] = inputs;

  if (typeof a.value !== "number" || typeof b.value !== "number") {
    throw new TypeError("COMPARE requires number inputs");
  }

  let result: number;
  if (a.value < b.value) {
    result = -1;
  } else if (a.value > b.value) {
    result = 1;
  } else {
    result = 0;
  }

  return {
    kind: "natural",
    value: result
  };
}
