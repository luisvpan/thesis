import * as rational from "../runtime/rational";
import type { RuntimeValue } from "../runtime/types";
import { RuntimeError } from "../runtime/errors";
import { flattenArrays, getComparableValue } from "./utils";

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

  return { kind: "arreglo", elements: filtered };
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

  return { kind: "arreglo", elements: filtered };
}
