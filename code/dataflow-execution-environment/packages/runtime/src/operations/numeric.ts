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

function isBoolean(value: unknown): value is Boolean {
  return typeof value === "object" && value !== null && "kind" in value && (value as { kind: string }).kind === "boolean";
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

function addIntegers(a: Integer, b: Integer): Integer {
  const aVal = a.value;
  const bVal = b.value;
  
  return {
    kind: "integer",
    value: aVal + bVal
  };
}

function addDecimals(a: Decimal, b: Decimal): Decimal {
  const aVal = a.value;
  const bVal = b.value;
  
  if (isNaN(aVal) || isNaN(bVal) || isNaN(aVal + bVal)) {
    throw new Error("ADD: Invalid operation (NaN result)");
  }
  
  return {
    kind: "decimal",
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

function subtractIntegers(a: Integer, b: Integer): Integer {
  const aVal = a.value;
  const bVal = b.value;
  
  return {
    kind: "integer",
    value: aVal - bVal
  };
}

function subtractDecimals(a: Decimal, b: Decimal): Decimal {
  const aVal = a.value;
  const bVal = b.value;
  
  if (isNaN(aVal) || isNaN(bVal) || isNaN(aVal - bVal)) {
    throw new Error("SUBTRACT: Invalid operation (NaN result)");
  }
  
  return {
    kind: "decimal",
    value: aVal - bVal
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

function multiplyIntegers(a: Integer, b: Integer): Integer {
  const aVal = a.value;
  const bVal = b.value;
  
  return {
    kind: "integer",
    value: aVal * bVal
  };
}

function multiplyDecimals(a: Decimal, b: Decimal): Decimal {
  const aVal = a.value;
  const bVal = b.value;
  
  if (isNaN(aVal) || isNaN(bVal) || isNaN(aVal * bVal)) {
    throw new Error("MULTIPLY: Invalid operation (NaN result)");
  }
  
  return {
    kind: "decimal",
    value: aVal * bVal
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

function divideIntegers(a: Integer, b: Integer): Decimal {
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

function divideDecimals(a: Decimal, b: Decimal): Decimal {
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

function compareIntegers(a: Integer, b: Integer): Boolean {
  const aVal = a.value;
  const bVal = b.value;

  return {
    kind: "boolean",
    value: aVal === bVal
  };
}

function compareDecimals(a: Decimal, b: Decimal): Boolean {
  const aVal = a.value;
  const bVal = b.value;

  return {
    kind: "boolean",
    value: aVal === bVal
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

function compareText(a: { kind: string; value: string }, b: { kind: string; value: string }): Boolean {
  return {
    kind: "boolean",
    value: a.value === b.value
  };
}

function compareBoolean(a: Boolean, b: Boolean): Boolean {
  return {
    kind: "boolean",
    value: a.value === b.value
  };
}

export function ADD(inputs: Array<{ id: string; value: unknown }>): Natural | Integer | Decimal | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return addNaturals(a.value, b.value);
  }
  if (isInteger(a.value) && isInteger(b.value)) {
    return addIntegers(a.value, b.value);
  }
  if (isDecimal(a.value) && isDecimal(b.value)) {
    return addDecimals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return addFractions(a.value, b.value);
  }
  if (isNatural(a.value) && isInteger(b.value)) {
    return addIntegers({ kind: "integer", value: a.value.value }, b.value);
  }
  if (isInteger(a.value) && isNatural(b.value)) {
    return addIntegers(a.value, { kind: "integer", value: b.value.value });
  }
  if (isNatural(a.value) && isDecimal(b.value)) {
    return addDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isNatural(b.value)) {
    return addDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  if (isInteger(a.value) && isDecimal(b.value)) {
    return addDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isInteger(b.value)) {
    return addDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  
  throw new TypeError(`ADD: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function SUBTRACT(inputs: Array<{ id: string; value: unknown }>): Integer | Decimal | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return subtractNaturals(a.value, b.value);
  }
  if (isInteger(a.value) && isInteger(b.value)) {
    return subtractIntegers(a.value, b.value);
  }
  if (isDecimal(a.value) && isDecimal(b.value)) {
    return subtractDecimals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return subtractFractions(a.value, b.value);
  }
  if (isNatural(a.value) && isInteger(b.value)) {
    return subtractIntegers({ kind: "integer", value: a.value.value }, b.value);
  }
  if (isInteger(a.value) && isNatural(b.value)) {
    return subtractIntegers(a.value, { kind: "integer", value: b.value.value });
  }
  if (isNatural(a.value) && isDecimal(b.value)) {
    return subtractDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isNatural(b.value)) {
    return subtractDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  if (isInteger(a.value) && isDecimal(b.value)) {
    return subtractDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isInteger(b.value)) {
    return subtractDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  
  throw new TypeError(`SUBTRACT: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function MULTIPLY(inputs: Array<{ id: string; value: unknown }>): Natural | Integer | Decimal | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return multiplyNaturals(a.value, b.value);
  }
  if (isNatural(a.value) && isInteger(b.value)) {
    const result = multiplyIntegers({ kind: "integer", value: a.value.value }, b.value);
    if (result.value >= 0) {
      return { kind: "natural", value: result.value };
    }
    return result;
  }
  if (isInteger(a.value) && isNatural(b.value)) {
    const result = multiplyIntegers(a.value, { kind: "integer", value: b.value.value });
    if (result.value >= 0) {
      return { kind: "natural", value: result.value };
    }
    return result;
  }
  if (isInteger(a.value) && isInteger(b.value)) {
    const result = multiplyIntegers(a.value, b.value);
    if (result.value >= 0) {
      return { kind: "natural", value: result.value };
    }
    return result;
  }
  if (isDecimal(a.value) && isDecimal(b.value)) {
    return multiplyDecimals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return multiplyFractions(a.value, b.value);
  }
  if (isNatural(a.value) && isDecimal(b.value)) {
    return multiplyDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isNatural(b.value)) {
    return multiplyDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  if (isInteger(a.value) && isDecimal(b.value)) {
    return multiplyDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isInteger(b.value)) {
    return multiplyDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  
  throw new TypeError(`MULTIPLY: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function DIVIDE(inputs: Array<{ id: string; value: unknown }>): Decimal | Fraction {
  const [a, b] = inputs;
  
  if (isNatural(a.value) && isNatural(b.value)) {
    return divideNaturals(a.value, b.value);
  }
  if (isInteger(a.value) && isInteger(b.value)) {
    return divideIntegers(a.value, b.value);
  }
  if (isDecimal(a.value) && isDecimal(b.value)) {
    return divideDecimals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return divideFractions(a.value, b.value);
  }
  if (isNatural(a.value) && isInteger(b.value)) {
    return divideIntegers({ kind: "integer", value: a.value.value }, b.value);
  }
  if (isInteger(a.value) && isNatural(b.value)) {
    return divideIntegers(a.value, { kind: "integer", value: b.value.value });
  }
  if (isNatural(a.value) && isDecimal(b.value)) {
    return divideDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isNatural(b.value)) {
    return divideDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  if (isInteger(a.value) && isDecimal(b.value)) {
    return divideDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isInteger(b.value)) {
    return divideDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  
  throw new TypeError(`DIVIDE: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}

export function COMPARE(inputs: Array<{ id: string; value: unknown }>): Boolean {
  const [a, b] = inputs;

  if (isNatural(a.value) && isNatural(b.value)) {
    return compareNaturals(a.value, b.value);
  }
  if (isInteger(a.value) && isInteger(b.value)) {
    return compareIntegers(a.value, b.value);
  }
  if (isDecimal(a.value) && isDecimal(b.value)) {
    return compareDecimals(a.value, b.value);
  }
  if (isFraction(a.value) && isFraction(b.value)) {
    return compareFractions(a.value, b.value);
  }
  if (typeof a.value === "object" && a.value !== null && "kind" in a.value && (a.value as { kind: string }).kind === "text") {
    if (typeof b.value === "object" && b.value !== null && "kind" in b.value && (b.value as { kind: string }).kind === "text") {
      return compareText(a.value as { kind: string; value: string }, b.value as { kind: string; value: string });
    }
  }
  if (isBoolean(a.value) && isBoolean(b.value)) {
    return compareBoolean(a.value, b.value);
  }
  if (isNatural(a.value) && isInteger(b.value)) {
    return compareIntegers({ kind: "integer", value: a.value.value }, b.value);
  }
  if (isInteger(a.value) && isNatural(b.value)) {
    return compareIntegers(a.value, { kind: "integer", value: b.value.value });
  }
  if (isNatural(a.value) && isDecimal(b.value)) {
    return compareDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isNatural(b.value)) {
    return compareDecimals(a.value, { kind: "decimal", value: b.value.value });
  }
  if (isInteger(a.value) && isDecimal(b.value)) {
    return compareDecimals({ kind: "decimal", value: a.value.value }, b.value);
  }
  if (isDecimal(a.value) && isInteger(b.value)) {
    return compareDecimals(a.value, { kind: "decimal", value: b.value.value });
  }

  throw new TypeError(`COMPARE: Unsupported types ${JSON.stringify(a.value)}, ${JSON.stringify(b.value)}`);
}
