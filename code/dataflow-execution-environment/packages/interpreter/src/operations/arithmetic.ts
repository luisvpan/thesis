import type Fraction from "fraction.js";
import * as rational from "../runtime/rational";
import type {
  RuntimeValue,
  RationalValue,
  CPAObject,
} from "../runtime/types";
import {
  isRational,
  isCPAObject,
  getCPAKey,
} from "../runtime/types";
import { RuntimeError } from "../runtime/errors";
import { flattenArrays, getQuantity, cloneCPAWithQuantity } from "./utils";

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
    return { kind: "racional", value: total };
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
    resultElements.push({ kind: "racional", value: rationalSum });
  }

  if (resultElements.length === 1) {
    return resultElements[0];
  }

  return { kind: "arreglo", elements: resultElements };
}

/**
 * Multiply operation (variadic):
 * - Flatten all input arrays
 * - For rationals only: multiply all values
 * - For CPA objects: apply rational factor to each CPA individually
 *
 * DESIGN NOTE (Scratch-style, Plan C): CPA objects are NEVER combined with
 * each other. Multiplying two CPAs (e.g., cap × cap) is semantically ambiguous
 * (what is "caps²"?). Instead, CPAs are returned as-is in an array, with any
 * rational factors applied to each individually. Visual blocking will be
 * enforced in the frontend.
 */
export function multiply(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // If all rationals, multiply them
  if (flatValues.every(isRational)) {
    const total = flatValues.reduce(
      (acc, v) => rational.multiply(acc, (v as RationalValue).value),
      rational.one()
    );
    return { kind: "racional", value: total };
  }

  // Separate CPAs by category: abstractos act as scaling factors, others are objects to scale
  const nonAbstractCPAs: CPAObject[] = [];
  const scalingFactors: Fraction[] = [];

  for (const value of flatValues) {
    if (isRational(value)) {
      scalingFactors.push(value.value);
    } else if (isCPAObject(value)) {
      if (value.category === "abstracto") {
        // CPA abstractos (numbers) act as scaling factors
        scalingFactors.push(getQuantity(value));
      } else {
        nonAbstractCPAs.push(value);
      }
    }
  }

  // If all values were scaling factors (rationals or abstractos), multiply them
  if (nonAbstractCPAs.length === 0) {
    const total = scalingFactors.reduce(
      (acc, v) => rational.multiply(acc, v),
      rational.one()
    );
    // Return as CPA abstracto if there were any abstractos, otherwise as racional
    const abstracto = flatValues.find(v => isCPAObject(v) && (v as CPAObject).category === "abstracto") as CPAObject | undefined;
    if (abstracto) {
      return cloneCPAWithQuantity(abstracto, total);
    }
    return { kind: "racional", value: total };
  }

  // Calculate the product of all scaling factors (1 if none)
  const scalingProduct = scalingFactors.length > 0
    ? scalingFactors.reduce((acc, v) => rational.multiply(acc, v), rational.one())
    : rational.one();

  // Apply the scaling factor to each non-abstract CPA individually (NO combining)
  const scaledCPAs = nonAbstractCPAs.map(cpa =>
    cloneCPAWithQuantity(cpa, rational.multiply(getQuantity(cpa), scalingProduct))
  );

  // If there's a single CPA, return it directly
  if (scaledCPAs.length === 1) {
    return scaledCPAs[0];
  }

  // Multiple CPAs: return array
  return { kind: "arreglo", elements: scaledCPAs };
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
      kind: "racional",
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
      kind: "racional",
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
