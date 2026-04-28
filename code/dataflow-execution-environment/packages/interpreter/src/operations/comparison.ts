import type Fraction from "fraction.js";
import * as rational from "../runtime/rational";
import type { RuntimeValue, RationalValue, CPAObject } from "../runtime/types";
import { isRational, isArray, isCPAObject } from "../runtime/types";
import { RuntimeError } from "../runtime/errors";

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
 * Flattens nested arrays recursively.
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
 * Gets the comparable value from a runtime value.
 */
function getComparableValue(val: RuntimeValue): Fraction | null {
  if (isRational(val)) {
    return val.value;
  }
  if (isCPAObject(val)) {
    return getQuantity(val);
  }
  return null;
}

/**
 * Less than operation (variadic):
 * - Last argument is the threshold
 * - Filters items where amount/value < threshold
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
  const filtered = flatItems.filter((item) => {
    const itemValue = getComparableValue(item);
    if (itemValue === null) return false;
    return rational.lessThan(itemValue, thresholdValue);
  });

  if (filtered.length === 1) {
    return filtered[0];
  }

  return { kind: "array", elements: filtered };
}

/**
 * Greater than operation (variadic):
 * - Last argument is the threshold
 * - Filters items where amount/value > threshold
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
  const filtered = flatItems.filter((item) => {
    const itemValue = getComparableValue(item);
    if (itemValue === null) return false;
    return rational.greaterThan(itemValue, thresholdValue);
  });

  if (filtered.length === 1) {
    return filtered[0];
  }

  return { kind: "array", elements: filtered };
}
