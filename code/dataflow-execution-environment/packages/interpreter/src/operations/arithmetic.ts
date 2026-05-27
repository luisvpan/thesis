import type Fraction from "fraction.js";
import * as rational from "../runtime/rational";
import type {
  RuntimeValue,
  CPAObject,
  CriteriaObject,
} from "../runtime/types";
import {
  isCPAObject,
  isArray,
  isCriteria,
  getCPAKey,
} from "../runtime/types";
import { RuntimeError } from "../runtime/errors";
import { flattenArrays, getQuantity, cloneCPAWithQuantity } from "./utils";
import { createAbstractNumber } from "../utils";

/**
 * Extracts all criteria from a flattened array of values.
 * Used to pass-through criteria in arithmetic operations.
 */
function extractCriteria(values: RuntimeValue[]): CriteriaObject[] {
  return values.filter(isCriteria);
}

/**
 * Filters out criteria from values, keeping only non-criteria values.
 */
function filterOutCriteria(values: RuntimeValue[]): RuntimeValue[] {
  return values.filter(v => !isCriteria(v));
}

/**
 * Appends criteria to a result, returning array if needed.
 * If criteria exist, always returns array: [...result, ...criteria]
 */
function appendCriteriaToResult(
  result: RuntimeValue,
  criteria: CriteriaObject[]
): RuntimeValue {
  if (criteria.length === 0) {
    return result;
  }

  // Convert result to array of elements
  const resultElements: RuntimeValue[] = isArray(result)
    ? result.elements
    : [result];

  return {
    kind: "arreglo",
    elements: [...resultElements, ...criteria],
  };
}

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
 * - Criteria are passed through to the end of the result (v4.0.0)
 */
export function sum(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // Extract and pass-through criteria (v4.0.0)
  const criteria = extractCriteria(flatValues);
  const dataValues = filterOutCriteria(flatValues);

  // If all abstract numbers, sum them and return abstract number
  if (dataValues.every(isAbstractNumber)) {
    const total = dataValues.reduce(
      (acc, v) => rational.add(acc, getQuantity(v as CPAObject)),
      rational.zero()
    );
    return appendCriteriaToResult(createAbstractNumber(total), criteria);
  }

  // CPA aggregation: group by key, sum quantities
  const groups = new Map<string, CPAObject>();

  for (const value of dataValues) {
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

  let result: RuntimeValue;
  if (resultElements.length === 1) {
    result = resultElements[0];
  } else {
    result = { kind: "arreglo", elements: resultElements };
  }

  return appendCriteriaToResult(result, criteria);
}

/**
 * Multiply operation (variadic):
 * - Flatten all input arrays
 * - For abstract numbers only: multiply all values
 * - For CPA objects: apply scaling factor to each CPA individually
 * - Criteria are passed through to the end of the result (v4.0.0)
 *
 * DESIGN NOTE (Scratch-style, Plan C): CPA objects are NEVER combined with
 * each other. Multiplying two CPAs (e.g., cap × cap) is semantically ambiguous
 * (what is "caps²"?). Instead, CPAs are returned as-is in an array, with any
 * scaling factors applied to each individually. Visual blocking will be
 * enforced in the frontend.
 */
export function multiply(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  // Extract and pass-through criteria (v4.0.0)
  const criteria = extractCriteria(flatValues);
  const dataValues = filterOutCriteria(flatValues);

  // If all abstract numbers, multiply them
  if (dataValues.every(isAbstractNumber)) {
    const total = dataValues.reduce(
      (acc, v) => rational.multiply(acc, getQuantity(v as CPAObject)),
      rational.one()
    );
    return appendCriteriaToResult(createAbstractNumber(total), criteria);
  }

  // Separate CPAs by category: abstractos act as scaling factors, others are objects to scale
  const nonAbstractCPAs: CPAObject[] = [];
  const scalingFactors: Fraction[] = [];

  for (const value of dataValues) {
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
    return appendCriteriaToResult(createAbstractNumber(total), criteria);
  }

  // Calculate the product of all scaling factors (1 if none)
  const scalingProduct = scalingFactors.length > 0
    ? scalingFactors.reduce((acc, v) => rational.multiply(acc, v), rational.one())
    : rational.one();

  // Apply the scaling factor to each non-abstract CPA individually (NO combining)
  const scaledCPAs = nonAbstractCPAs.map(cpa =>
    cloneCPAWithQuantity(cpa, rational.multiply(getQuantity(cpa), scalingProduct))
  );

  let result: RuntimeValue;
  // If there's a single CPA, return it directly
  if (scaledCPAs.length === 1) {
    result = scaledCPAs[0];
  } else {
    // Multiple CPAs: return array
    result = { kind: "arreglo", elements: scaledCPAs };
  }

  return appendCriteriaToResult(result, criteria);
}

/**
 * Helper to extract CPAObjects from a RuntimeValue (flattening arrays)
 */
function extractCPAsFromValue(val: RuntimeValue): CPAObject[] {
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
 * Helper to extract CriteriaObjects from a RuntimeValue (flattening arrays)
 */
function extractCriteriaFromValue(val: RuntimeValue): CriteriaObject[] {
  if (isCriteria(val)) {
    return [val];
  }
  if (isArray(val)) {
    const flat = flattenArrays(val.elements);
    return flat.filter(isCriteria);
  }
  return [];
}

/**
 * Substract operation (binary, arity = 2):
 * - Groups both arguments by CPA key
 * - Computes difference per key: quantity_a - quantity_b
 * - Keys only in b result in negative quantities
 * - Returns array with all results (or single CPA if only one)
 * - Criteria are passed through to the end of the result (v4.0.0)
 */
export function substract(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `substract requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [rawA, rawB] = args;

  // Extract and pass-through criteria from both arguments (v4.0.0)
  const criteriaA = extractCriteriaFromValue(rawA);
  const criteriaB = extractCriteriaFromValue(rawB);
  const criteria = [...criteriaA, ...criteriaB];

  const cpasA = extractCPAsFromValue(rawA);
  const cpasB = extractCPAsFromValue(rawB);

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

  let result: RuntimeValue;
  if (results.length === 0) {
    result = { kind: "arreglo", elements: [] };
  } else if (results.length === 1) {
    result = results[0];
  } else {
    result = { kind: "arreglo", elements: results };
  }

  return appendCriteriaToResult(result, criteria);
}

/**
 * Divide operation (binary, arity = 2):
 * - Extracts only abstract numbers from b and sums them as divisor
 * - If no abstract numbers in b, divisor is implicitly 1
 * - Aggregates a by CPA key (implicit sum), then divides each group by the divisor
 * - Non-number CPAs in b are ignored
 * - Criteria are passed through to the end of the result (v4.0.0)
 */
export function divide(args: RuntimeValue[]): RuntimeValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `divide requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [rawA, rawB] = args;

  // Extract and pass-through criteria from both arguments (v4.0.0)
  const criteriaA = extractCriteriaFromValue(rawA);
  const criteriaB = extractCriteriaFromValue(rawB);
  const criteria = [...criteriaA, ...criteriaB];

  const cpasA = extractCPAsFromValue(rawA);
  const cpasB = extractCPAsFromValue(rawB);

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

  let result: RuntimeValue;
  if (results.length === 0) {
    result = { kind: "arreglo", elements: [] };
  } else if (results.length === 1) {
    result = results[0];
  } else {
    result = { kind: "arreglo", elements: results };
  }

  return appendCriteriaToResult(result, criteria);
}
