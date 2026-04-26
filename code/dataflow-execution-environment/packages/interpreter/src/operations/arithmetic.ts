import Fraction from "fraction.js";
import * as rational from "../runtime/rational";
import type {
  RuntimeValue,
  RationalValue,
  ArrayValue,
  CPAObject,
  ShapeValue,
  FoodValue,
  AbstractValue,
} from "../runtime/types";
import {
  isRational,
  isArray,
  isCPAObject,
  getCPAKey,
} from "../runtime/types";
import { RuntimeError } from "../runtime/errors";

/**
 * Flattens nested arrays recursively into a single array of values.
 */
function flattenArrays(values: RuntimeValue[]): RuntimeValue[] {
  const result: RuntimeValue[] = [];

  for (const val of values) {
    if (isArray(val)) {
      result.push(...flattenArrays(val.elements));
    } else {
      result.push(val);
    }
  }

  return result;
}

/**
 * Clones a CPA object with a new amount/value.
 */
function cloneCPAWithQuantity(obj: CPAObject, quantity: Fraction): CPAObject {
  if (obj.kind === "abstract") {
    return { ...obj, value: quantity };
  }
  if (obj.kind === "shape") {
    return { ...obj, amount: quantity };
  }
  if (obj.kind === "food") {
    return { ...obj, amount: quantity };
  }
  return obj;
}

/**
 * Gets the quantity (value for abstract, amount for pictorial/concrete).
 */
function getQuantity(obj: CPAObject): Fraction {
  if (obj.kind === "abstract") {
    return obj.value;
  }
  return obj.amount;
}

/**
 * Sum operation (variadic):
 * - Flatten all input arrays
 * - For rationals: add all values
 * - For CPA objects: aggregate by (category+type+subtype), sum amounts
 */
export function sum(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // If all rationals, sum them
  if (flatValues.every(isRational)) {
    const total = flatValues.reduce(
      (acc, v) => rational.add(acc, (v as RationalValue).value),
      rational.zero()
    );
    return { kind: "rational", value: total };
  }

  // CPA aggregation: group by key, sum quantities
  const groups = new Map<string, CPAObject>();
  const rationals: Fraction[] = [];

  for (const value of flatValues) {
    if (isRational(value)) {
      rationals.push(value.value);
    } else if (isCPAObject(value)) {
      const key = getCPAKey(value);
      const existing = groups.get(key);

      if (existing) {
        const newQty = rational.add(getQuantity(existing), getQuantity(value));
        groups.set(key, cloneCPAWithQuantity(existing, newQty));
      } else {
        groups.set(key, cloneCPAWithQuantity(value, getQuantity(value)));
      }
    }
  }

  // If we have both rationals and CPA objects, return array
  const resultElements: RuntimeValue[] = Array.from(groups.values());

  if (rationals.length > 0) {
    const rationalSum = rationals.reduce(
      (acc, v) => rational.add(acc, v),
      rational.zero()
    );
    resultElements.push({ kind: "rational", value: rationalSum });
  }

  if (resultElements.length === 1) {
    return resultElements[0];
  }

  return { kind: "array", elements: resultElements };
}

/**
 * Multiply operation (variadic):
 * - Flatten all input arrays
 * - For rationals: multiply all values
 * - For CPA objects: aggregate by key, multiply amounts
 */
export function multiply(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // If all rationals, multiply them
  if (flatValues.every(isRational)) {
    const total = flatValues.reduce(
      (acc, v) => rational.multiply(acc, (v as RationalValue).value),
      rational.one()
    );
    return { kind: "rational", value: total };
  }

  // CPA aggregation: group by key, multiply quantities
  const groups = new Map<string, CPAObject>();
  const rationals: Fraction[] = [];

  for (const value of flatValues) {
    if (isRational(value)) {
      rationals.push(value.value);
    } else if (isCPAObject(value)) {
      const key = getCPAKey(value);
      const existing = groups.get(key);

      if (existing) {
        const newQty = rational.multiply(getQuantity(existing), getQuantity(value));
        groups.set(key, cloneCPAWithQuantity(existing, newQty));
      } else {
        groups.set(key, cloneCPAWithQuantity(value, getQuantity(value)));
      }
    }
  }

  // Apply rational multiplier to all CPA objects
  if (rationals.length > 0) {
    const rationalProduct = rationals.reduce(
      (acc, v) => rational.multiply(acc, v),
      rational.one()
    );

    for (const [key, obj] of groups) {
      const newQty = rational.multiply(getQuantity(obj), rationalProduct);
      groups.set(key, cloneCPAWithQuantity(obj, newQty));
    }
  }

  const resultElements = Array.from(groups.values());

  if (resultElements.length === 0 && rationals.length > 0) {
    const rationalProduct = rationals.reduce(
      (acc, v) => rational.multiply(acc, v),
      rational.one()
    );
    return { kind: "rational", value: rationalProduct };
  }

  if (resultElements.length === 1) {
    return resultElements[0];
  }

  return { kind: "array", elements: resultElements };
}

/**
 * Substract operation (binary, arity = 2):
 * - a - b for amounts/values
 */
export function substract(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `substract requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [a, b] = args;

  // Rational - Rational
  if (isRational(a) && isRational(b)) {
    return {
      kind: "rational",
      value: rational.subtract(a.value, b.value),
    };
  }

  // CPA - CPA (subtract quantities)
  if (isCPAObject(a) && isCPAObject(b)) {
    const diff = rational.subtract(getQuantity(a), getQuantity(b));
    return cloneCPAWithQuantity(a, diff);
  }

  // CPA - Rational (subtract from quantity)
  if (isCPAObject(a) && isRational(b)) {
    const diff = rational.subtract(getQuantity(a), b.value);
    return cloneCPAWithQuantity(a, diff);
  }

  throw new RuntimeError(
    "TYPE_ERROR",
    "substract requires compatible numeric types"
  );
}

/**
 * Divide operation (binary, arity = 2):
 * - a / b for amounts/values
 */
export function divide(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `divide requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [a, b] = args;

  // Get divisor value
  let divisor: Fraction;
  if (isRational(b)) {
    divisor = b.value;
  } else if (isCPAObject(b)) {
    divisor = getQuantity(b);
  } else {
    throw new RuntimeError("TYPE_ERROR", "divide requires numeric divisor");
  }

  // Rational / Rational
  if (isRational(a)) {
    return {
      kind: "rational",
      value: rational.divide(a.value, divisor),
    };
  }

  // CPA / value (divide quantity)
  if (isCPAObject(a)) {
    const quotient = rational.divide(getQuantity(a), divisor);
    return cloneCPAWithQuantity(a, quotient);
  }

  throw new RuntimeError(
    "TYPE_ERROR",
    "divide requires compatible numeric types"
  );
}
