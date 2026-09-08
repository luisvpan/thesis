import type { RuntimeValue } from "../runtime/types";
import { isArray } from "../runtime/types";

/**
 * Returns the first element of an array.
 * - If no arguments: returns empty array
 * - If argument is not an array: returns the argument itself
 * - If array is empty: returns empty array (no error)
 */
export function first(args: RuntimeValue[]): RuntimeValue {
  if (args.length === 0) {
    return { kind: "arreglo", elements: [] };
  }

  const val = args[0];

  if (!isArray(val)) {
    return val;
  }

  if (val.elements.length === 0) {
    return { kind: "arreglo", elements: [] };
  }

  return val.elements[0];
}

/**
 * Returns the last element of an array.
 * - If no arguments: returns empty array
 * - If argument is not an array: returns the argument itself
 * - If array is empty: returns empty array (no error)
 */
export function last(args: RuntimeValue[]): RuntimeValue {
  if (args.length === 0) {
    return { kind: "arreglo", elements: [] };
  }

  const val = args[0];

  if (!isArray(val)) {
    return val;
  }

  if (val.elements.length === 0) {
    return { kind: "arreglo", elements: [] };
  }

  return val.elements[val.elements.length - 1];
}
