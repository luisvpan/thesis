import type Fraction from "fraction.js";
import * as rational from "../runtime/rational";
import type {
  RuntimeValue,
  CPAObject,
} from "../runtime/types";
import {
  isCPAObject,
  isArray,
  getCPAKey,
} from "../runtime/types";
import { RuntimeError } from "../runtime/errors";
import { flattenArrays, getQuantity, cloneCPAWithQuantity } from "./utils";
import { createAbstractNumber } from "../utils";

/**
 * Check if a value is an abstract number (CPA abstracto with type "numero")
 */
function isAbstractNumber(val: RuntimeValue): val is CPAObject {
  return isCPAObject(val) && val.category === "abstracto" && val.type === "numero";
}

/**
 * Sum operation (variadic):
 * - Flatten all input arrays
 * - For abstract numbers: add all values
 * - For CPA objects: aggregate by (category+type+subtype), sum amounts
 */
export function sum(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // If all abstract numbers, sum them and return abstract number
  if (flatValues.every(isAbstractNumber)) {
    const total = flatValues.reduce(
      (acc, v) => rational.add(acc, getQuantity(v as CPAObject)),
      rational.zero()
    );
    return createAbstractNumber(total);
  }

  // CPA aggregation: group by key, sum quantities
  const groups = new Map<string, CPAObject>();

  for (const value of flatValues) {
    if (isCPAObject(value)) {
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

  const resultElements: RuntimeValue[] = Array.from(groups.values());

  if (resultElements.length === 1) {
    return resultElements[0];
  }

  return { kind: "arreglo", elements: resultElements };
}

/**
 * Multiply operation (variadic):
 * - Flatten all input arrays
 * - For abstract numbers only: multiply all values
 * - For CPA objects: apply scaling factor to each CPA individually
 *
 * DESIGN NOTE (Scratch-style, Plan C): CPA objects are NEVER combined with
 * each other. Multiplying two CPAs (e.g., cap × cap) is semantically ambiguous
 * (what is "caps²"?). Instead, CPAs are returned as-is in an array, with any
 * scaling factors applied to each individually. Visual blocking will be
 * enforced in the frontend.
 */
export function multiply(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // If all abstract numbers, multiply them
  if (flatValues.every(isAbstractNumber)) {
    const total = flatValues.reduce(
      (acc, v) => rational.multiply(acc, getQuantity(v as CPAObject)),
      rational.one()
    );
    return createAbstractNumber(total);
  }

  // Separate CPAs by category: abstractos act as scaling factors, others are objects to scale
  const nonAbstractCPAs: CPAObject[] = [];
  const scalingFactors: Fraction[] = [];

  for (const value of flatValues) {
    if (isCPAObject(value)) {
      if (value.category === "abstracto") {
        // CPA abstractos (numbers) act as scaling factors
        scalingFactors.push(getQuantity(value));
      } else {
        nonAbstractCPAs.push(value);
      }
    }
  }

  // If all values were scaling factors (abstractos), multiply them
  if (nonAbstractCPAs.length === 0) {
    const total = scalingFactors.reduce(
      (acc, v) => rational.multiply(acc, v),
      rational.one()
    );
    return createAbstractNumber(total);
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
 * Resolves a RuntimeValue to a CPAObject for binary arithmetic operations.
 * - CPAObject: return as-is
 * - ArrayValue: implicit sum of elements, return resulting CPAObject
 * - OtherValue: return null (ignored)
 */
function resolveToSingleCPA(val: RuntimeValue): CPAObject | null {
  if (isCPAObject(val)) {
    return val;
  }
  if (isArray(val)) {
    // Implicit sum of array elements
    const result = sum(val.elements);
    if (isCPAObject(result)) {
      return result;
    }
    // If sum returns an array (heterogeneous), take first element
    if (isArray(result) && result.elements.length > 0 && isCPAObject(result.elements[0])) {
      return result.elements[0] as CPAObject;
    }
    return null;
  }
  // OtherValue - ignored
  return null;
}

/**
 * Substract operation (binary, arity = 2):
 * - a - b for amounts/values
 * - Supports ArrayValue (implicit sum) and OtherValue (ignored)
 */
export function substract(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `substract requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [rawA, rawB] = args;
  const a = resolveToSingleCPA(rawA);
  const b = resolveToSingleCPA(rawB);

  // If both are null (OtherValue), return first arg as-is
  if (a === null && b === null) {
    return rawA;
  }
  // If a is null, return b as-is
  if (a === null) {
    return b!;
  }
  // If b is null, return a as-is
  if (b === null) {
    return a;
  }

  const diff = rational.subtract(getQuantity(a), getQuantity(b));
  return cloneCPAWithQuantity(a, diff);
}

/**
 * Divide operation (binary, arity = 2):
 * - a / b for amounts/values
 * - Supports ArrayValue (implicit sum) and OtherValue (ignored)
 */
export function divide(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `divide requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [rawA, rawB] = args;
  const a = resolveToSingleCPA(rawA);
  const b = resolveToSingleCPA(rawB);

  // If both are null (OtherValue), return first arg as-is
  if (a === null && b === null) {
    return rawA;
  }
  // If a is null, return b as-is
  if (a === null) {
    return b!;
  }
  // If b is null, return a as-is (can't divide by nothing)
  if (b === null) {
    return a;
  }

  const divisor = getQuantity(b);
  const quotient = rational.divide(getQuantity(a), divisor);
  return cloneCPAWithQuantity(a, quotient);
}
