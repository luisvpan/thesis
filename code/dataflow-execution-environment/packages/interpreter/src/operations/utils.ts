import type Fraction from "fraction.js";
import type { RuntimeValue, CPAObject } from "../runtime/types";
import { isArray, isCPAObject } from "../runtime/types";
import * as rational from "../runtime/rational";

/**
 * Recursively flattens nested arrays into a single-level array.
 */
export function flattenArrays(values: RuntimeValue[]): RuntimeValue[] {
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
 * Gets the quantity from a CPA object.
 * All CPA objects now use the unified 'quantity' field.
 */
export function getQuantity(obj: CPAObject): Fraction {
  return obj.quantity;
}

/**
 * Gets the comparable numeric value from any RuntimeValue.
 * Returns null if the value is not comparable.
 */
export function getComparableValue(val: RuntimeValue): Fraction | null {
  if (isCPAObject(val)) {
    return getQuantity(val);
  }
  return null;
}

/**
 * Gets the quantity from any RuntimeValue, defaulting to zero.
 * Used for ordering operations.
 */
export function getQuantityOrZero(val: RuntimeValue): Fraction {
  if (isCPAObject(val)) {
    return getQuantity(val);
  }
  return rational.zero();
}

/**
 * Clones a CPA object with a new quantity.
 */
export function cloneCPAWithQuantity(obj: CPAObject, quantity: Fraction): CPAObject {
  return { ...obj, quantity };
}
