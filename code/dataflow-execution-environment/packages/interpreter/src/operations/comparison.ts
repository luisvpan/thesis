import * as rational from "../runtime/rational";
import type { RuntimeValue, CriteriaObject } from "../runtime/types";
import { isCriteria, isArray } from "../runtime/types";
import { RuntimeError } from "../runtime/errors";
import { flattenArrays, getComparableValue } from "./utils";

/**
 * Extracts all criteria from a flattened array of values.
 */
function extractCriteria(values: RuntimeValue[]): CriteriaObject[] {
  return values.filter(isCriteria);
}

/**
 * Filters out criteria from values.
 */
function filterOutCriteria(values: RuntimeValue[]): RuntimeValue[] {
  return values.filter(v => !isCriteria(v));
}

/**
 * Appends criteria to a result, returning array if needed.
 */
function appendCriteriaToResult(
  result: RuntimeValue,
  criteria: CriteriaObject[]
): RuntimeValue {
  if (criteria.length === 0) {
    return result;
  }

  const resultElements: RuntimeValue[] = isArray(result)
    ? result.elements
    : [result];

  return {
    kind: "arreglo",
    elements: [...resultElements, ...criteria],
  };
}

/**
 * Less than operation (variadic):
 * - Last argument is the threshold
 * - Filters items where amount/value < threshold
 * - Criteria are passed through to the end of the result (v4.0.0)
 */
export function lessThan(args: RuntimeValue[]): RuntimeValue {
  if (args.length < 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `less_than requires at least 2 arguments, got ${args.length}`
    );
  }

  // Last argument is the threshold
  const threshold = args[args.length - 1];
  const items = args.slice(0, -1);

  const thresholdValue = getComparableValue(threshold);
  if (thresholdValue === null) {
    throw new RuntimeError(
      "TYPE_ERROR",
      "less_than threshold must be a numeric value"
    );
  }

  // Flatten and filter
  const flatItems = flattenArrays(items);

  // Extract and pass-through criteria (v4.0.0)
  const criteria = extractCriteria(flatItems);
  const dataItems = filterOutCriteria(flatItems);

  const filtered = dataItems.filter((item) => {
    const itemValue = getComparableValue(item);
    if (itemValue === null) return false;
    return rational.lessThan(itemValue, thresholdValue);
  });

  let result: RuntimeValue;
  if (filtered.length === 1) {
    result = filtered[0];
  } else {
    result = { kind: "arreglo", elements: filtered };
  }

  return appendCriteriaToResult(result, criteria);
}

/**
 * Greater than operation (variadic):
 * - Last argument is the threshold
 * - Filters items where amount/value > threshold
 * - Criteria are passed through to the end of the result (v4.0.0)
 */
export function greaterThan(args: RuntimeValue[]): RuntimeValue {
  if (args.length < 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `greater_than requires at least 2 arguments, got ${args.length}`
    );
  }

  // Last argument is the threshold
  const threshold = args[args.length - 1];
  const items = args.slice(0, -1);

  const thresholdValue = getComparableValue(threshold);
  if (thresholdValue === null) {
    throw new RuntimeError(
      "TYPE_ERROR",
      "greater_than threshold must be a numeric value"
    );
  }

  // Flatten and filter
  const flatItems = flattenArrays(items);

  // Extract and pass-through criteria (v4.0.0)
  const criteria = extractCriteria(flatItems);
  const dataItems = filterOutCriteria(flatItems);

  const filtered = dataItems.filter((item) => {
    const itemValue = getComparableValue(item);
    if (itemValue === null) return false;
    return rational.greaterThan(itemValue, thresholdValue);
  });

  let result: RuntimeValue;
  if (filtered.length === 1) {
    result = filtered[0];
  } else {
    result = { kind: "arreglo", elements: filtered };
  }

  return appendCriteriaToResult(result, criteria);
}
