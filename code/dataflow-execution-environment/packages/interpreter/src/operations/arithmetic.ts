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
 * Helper to extract CPAObjects from a RuntimeValue (flattening arrays)
 */
function extractCPAs(val: RuntimeValue): CPAObject[] {
  if (isCPAObject(val)) {
    return [val];
  }
  if (isArray(val)) {
    const flat = flattenArrays(val.elements);
    return flat.filter(isCPAObject);
  }
  return [];
}

/**
 * Substract operation (binary, arity = 2):
 * - Groups both arguments by CPA key
 * - Computes difference per key: quantity_a - quantity_b
 * - Keys only in b result in negative quantities
 * - Returns array with all results (or single CPA if only one)
 */
export function substract(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `substract requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [rawA, rawB] = args;
  const cpasA = extractCPAs(rawA);
  const cpasB = extractCPAs(rawB);

  // Group by CPA key
  const groupsA = new Map<string, CPAObject>();
  const groupsB = new Map<string, CPAObject>();

  for (const val of cpasA) {
    const key = getCPAKey(val);
    const existing = groupsA.get(key);
    if (existing) {
      groupsA.set(key, cloneCPAWithQuantity(existing, rational.add(getQuantity(existing), getQuantity(val))));
    } else {
      groupsA.set(key, cloneCPAWithQuantity(val, getQuantity(val)));
    }
  }

  for (const val of cpasB) {
    const key = getCPAKey(val);
    const existing = groupsB.get(key);
    if (existing) {
      groupsB.set(key, cloneCPAWithQuantity(existing, rational.add(getQuantity(existing), getQuantity(val))));
    } else {
      groupsB.set(key, cloneCPAWithQuantity(val, getQuantity(val)));
    }
  }

  // Compute differences for all keys
  const allKeys = new Set([...groupsA.keys(), ...groupsB.keys()]);
  const results: CPAObject[] = [];

  for (const key of allKeys) {
    const objA = groupsA.get(key);
    const objB = groupsB.get(key);
    const qtyA = objA ? getQuantity(objA) : rational.zero();
    const qtyB = objB ? getQuantity(objB) : rational.zero();
    const diff = rational.subtract(qtyA, qtyB);

    // Use template from A if exists, otherwise from B
    const template = objA ?? objB!;
    results.push(cloneCPAWithQuantity(template, diff));
  }

  if (results.length === 0) {
    return { kind: "arreglo", elements: [] };
  }
  if (results.length === 1) {
    return results[0];
  }
  return { kind: "arreglo", elements: results };
}

/**
 * Divide operation (binary, arity = 2):
 * - Extracts only abstract numbers from b and sums them as divisor
 * - If no abstract numbers in b, divisor is implicitly 1
 * - Aggregates a by CPA key (implicit sum), then divides each group by the divisor
 * - Non-number CPAs in b are ignored
 */
export function divide(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `divide requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [rawA, rawB] = args;
  const cpasA = extractCPAs(rawA);
  const cpasB = extractCPAs(rawB);

  // Extract only abstract numbers from B and sum them
  let divisor = rational.zero();
  let hasAbstractNumber = false;

  for (const val of cpasB) {
    if (val.category === "abstracto" && val.type === "numero") {
      divisor = rational.add(divisor, getQuantity(val));
      hasAbstractNumber = true;
    }
  }

  // If no abstract numbers, divisor is implicitly 1
  if (!hasAbstractNumber) {
    divisor = rational.one();
  }

  // Check for division by zero
  if (divisor.equals(rational.zero())) {
    throw new RuntimeError("DIVISION_BY_ZERO", "Cannot divide by zero");
  }

  // Aggregate A by CPA key (implicit sum for same-key elements)
  const groupsA = new Map<string, CPAObject>();
  for (const val of cpasA) {
    const key = getCPAKey(val);
    const existing = groupsA.get(key);
    if (existing) {
      groupsA.set(key, cloneCPAWithQuantity(existing, rational.add(getQuantity(existing), getQuantity(val))));
    } else {
      groupsA.set(key, cloneCPAWithQuantity(val, getQuantity(val)));
    }
  }

  // Divide each aggregated group by the divisor
  const results: CPAObject[] = [];

  for (const cpa of groupsA.values()) {
    const quotient = rational.divide(getQuantity(cpa), divisor);
    results.push(cloneCPAWithQuantity(cpa, quotient));
  }

  if (results.length === 0) {
    return { kind: "arreglo", elements: [] };
  }
  if (results.length === 1) {
    return results[0];
  }
  return { kind: "arreglo", elements: results };
}
