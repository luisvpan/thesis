import type { Natural, Integer, Decimal, Boolean, Fraction, DataType } from "@dataflow/shared/types";

function unwrapValue(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "object" && value !== null && "value" in value) {
    return (value as { value: number }).value;
  }
  throw new TypeError("Expected number value");
}

function isNatural(value: unknown): value is Natural {
  return typeof value === "object" && value !== null && "kind" in value && (value as { kind: string }).kind === "natural";
}

function isInteger(value: unknown): value is Integer {
  return typeof value === "object" && value !== null && "kind" in value && (value as { kind: string }).kind === "integer";
}

function isDecimal(value: unknown): value is Decimal {
  return typeof value === "object" && value !== null && "kind" in value && (value as { kind: string }).kind === "decimal";
}

function isFraction(value: unknown): value is Fraction {
  return typeof value === "object" && value !== null && "kind" in value && (value as { kind: string }).kind === "fraction";
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

function addNaturals(a: Natural, b: Natural): Natural {
  const aVal = a.value;
  const bVal = b.value;
  
  if (aVal < 0 || bVal < 0) {
    throw new RangeError("ADD requires natural numbers (>= 0)");
  }
  
  return {
    kind: "natural",
    value: aVal + bVal
  };
}

function addFractions(a: Fraction, b: Fraction): Fraction {
  const f1 = unwrapFraction(a);
  const f2 = unwrapFraction(b);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("ADD: Denominator cannot be zero");
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

function subtractNaturals(a: Natural, b: Natural): Integer {
  const aVal = a.value;
  const bVal = b.value;
  
  return {
    kind: "integer",
    value: aVal - bVal
  };
}

function subtractFractions(a: Fraction, b: Fraction): Fraction {
  const f1 = unwrapFraction(a);
  const f2 = unwrapFraction(b);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("SUBTRACT: Denominator cannot be zero");
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

function multiplyNaturals(a: Natural, b: Natural): Natural {
  const aVal = a.value;
  const bVal = b.value;
  
  if (aVal < 0 || bVal < 0) {
    throw new RangeError("MULTIPLY requires natural numbers (>= 0)");
  }
  
  return {
    kind: "natural",
    value: aVal * bVal
  };
}

function multiplyFractions(a: Fraction, b: Fraction): Fraction {
  const f1 = unwrapFraction(a);
  const f2 = unwrapFraction(b);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("MULTIPLY: Denominator cannot be zero");
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

function divideNaturals(a: Natural, b: Natural): Decimal {
  const aVal = a.value;
  const bVal = b.value;
  
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

function divideFractions(a: Fraction, b: Fraction): Fraction {
  const f1 = unwrapFraction(a);
  const f2 = unwrapFraction(b);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("DIVIDE: Denominator cannot be zero");
  }

  const numerator = f1.numerator * f2.denominator;
  const denominator = f1.denominator * f2.numerator;

  if (denominator === 0) {
    throw new Error("DIVIDE: Division by zero");
  }

  const simplified = simplifyFraction(numerator, denominator);

  return {
    kind: "fraction",
    numerator: simplified.numerator,
    denominator: simplified.denominator
  };
}

function compareNaturals(a: Natural, b: Natural): Boolean {
  const aVal = a.value;
  const bVal = b.value;

  return {
    kind: "boolean",
    value: aVal === bVal
  };
}

function compareFractions(a: Fraction, b: Fraction): Boolean {
  const f1 = unwrapFraction(a);
  const f2 = unwrapFraction(b);

  if (f1.denominator === 0 || f2.denominator === 0) {
    throw new Error("COMPARE: Denominator cannot be zero");
  }

  const leftValue = f1.numerator / f1.denominator;
  const rightValue = f2.numerator / f2.denominator;

  return {
    kind: "boolean",
    value: leftValue === rightValue
  };
}

export function ADD(inputs: Array<{ id: string; value: unknown }>): Natural | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return addNaturals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return addFractions(a.value, b.value);
  }
  
  throw new TypeError(`ADD: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function SUBTRACT(inputs: Array<{ id: string; value: unknown }>): Integer | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return subtractNaturals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return subtractFractions(a.value, b.value);
  }
  
  throw new TypeError(`SUBTRACT: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function MULTIPLY(inputs: Array<{ id: string; value: unknown }>): Natural | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return multiplyNaturals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return multiplyFractions(a.value, b.value);
  }
  
  throw new TypeError(`MULTIPLY: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function DIVIDE(inputs: Array<{ id: string; value: unknown }>): Decimal | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return divideNaturals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return divideFractions(a.value, b.value);
  }
  
  throw new TypeError(`DIVIDE: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function COMPARE(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return compareNaturals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return compareFractions(a.value, b.value);
  }
  
  throw new TypeError(`COMPARE: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}
