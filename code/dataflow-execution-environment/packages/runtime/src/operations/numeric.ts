import type { Natural, Integer, Decimal, Boolean, Fraction } from "@dataflow/shared/types";

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
  
  if (isNaN(aVal) || isNaN(bVal) || isNaN(aVal / bVal)) {
    throw new Error("DIVIDE: Invalid operation (NaN result)");
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

function unwrapFraction(value: unknown): { numerator: number; denominator: number } {
  if (typeof value === "object" && value !== null && "kind" in value && (value as Fraction).kind === "fraction") {
    return {
      numerator: (value as Fraction).numerator,
      denominator: (value as Fraction).denominator
    };
  }
  throw new TypeError("Expected fraction value");
}

function gcd(a: number, b: number): number {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function simplifyFraction(numerator: number, denominator: number): { numerator: number; denominator: number } {
  if (denominator < 0) {
    numerator = -numerator;
    denominator = -denominator;
  }

  const common = gcd(numerator, denominator);
  return {
    numerator: numerator / common,
    denominator: denominator / common
  };
}

export function ADD_FRACTION(inputs: Array<{ id: string; value: unknown }>): Fraction {
  const [a, b] = inputs;
  const f1 = unwrapFraction(a.value);
  const f2 = unwrapFraction(b.value);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("ADD_FRACTION: Denominator cannot be zero");
  }

  const numerator = f1.numerator * f2.denominator + f2.numerator * f1.denominator;
  const denominator = f1.denominator * f2.denominator;

  const simplified = simplifyFraction(numerator, denominator);

  return {
    kind: "fraction",
    numerator: simplified.numerator,
    denominator: simplified.denominator
  };
}

export function SUBTRACT_FRACTION(inputs: Array<{ id: string; value: unknown }>): Fraction {
  const [a, b] = inputs;
  const f1 = unwrapFraction(a.value);
  const f2 = unwrapFraction(b.value);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("SUBTRACT_FRACTION: Denominator cannot be zero");
  }

  const numerator = f1.numerator * f2.denominator - f2.numerator * f1.denominator;
  const denominator = f1.denominator * f2.denominator;

  const simplified = simplifyFraction(numerator, denominator);

  return {
    kind: "fraction",
    numerator: simplified.numerator,
    denominator: simplified.denominator
  };
}

export function MULTIPLY_FRACTION(inputs: Array<{ id: string; value: unknown }>): Fraction {
  const [a, b] = inputs;
  const f1 = unwrapFraction(a.value);
  const f2 = unwrapFraction(b.value);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("MULTIPLY_FRACTION: Denominator cannot be zero");
  }

  const numerator = f1.numerator * f2.numerator;
  const denominator = f1.denominator * f2.denominator;

  const simplified = simplifyFraction(numerator, denominator);

  return {
    kind: "fraction",
    numerator: simplified.numerator,
    denominator: simplified.denominator
  };
}

export function DIVIDE_FRACTION(inputs: Array<{ id: string; value: unknown }>): Fraction {
  const [a, b] = inputs;
  const f1 = unwrapFraction(a.value);
  const f2 = unwrapFraction(b.value);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("DIVIDE_FRACTION: Denominator cannot be zero");
  }

  const numerator = f1.numerator * f2.denominator;
  const denominator = f1.denominator * f2.numerator;

  if (denominator === 0) {
    throw new Error("DIVIDE_FRACTION: Division by zero");
  }

  const simplified = simplifyFraction(numerator, denominator);

  return {
    kind: "fraction",
    numerator: simplified.numerator,
    denominator: simplified.denominator
  };
}

export function COMPARE_FRACTION(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [a, b] = inputs;
  const f1 = unwrapFraction(a.value);
  const f2 = unwrapFraction(b.value);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("COMPARE_FRACTION: Denominator cannot be zero");
  }

  const leftValue = f1.numerator / f1.denominator;
  const rightValue = f2.numerator / f2.denominator;

  return {
    kind: "boolean",
    value: leftValue === rightValue
  };
}
