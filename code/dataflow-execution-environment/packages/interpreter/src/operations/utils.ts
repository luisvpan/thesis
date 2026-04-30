import type Fraction from "fraction.js";
import type { RuntimeValue, CPAObject } from "../runtime/types";
import { isArray, isCPAObject, isRational } from "../runtime/types";
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
 * Gets the quantity/amount from a CPA object.
 * For abstracto: returns value
 * For forma/comida: returns amount
 */
export function getQuantity(obj: CPAObject): Fraction {
  if (obj.kind === "abstracto") {
    return obj.value;
  }
  return obj.amount;
}

/**
 * Gets the comparable numeric value from any RuntimeValue.
 * Returns null if the value is not comparable.
 */
export function getComparableValue(val: RuntimeValue): Fraction | null {
  if (isRational(val)) {
    return val.value;
  }
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
  if (isRational(val)) {
    return val.value;
  }
  if (isCPAObject(val)) {
    return getQuantity(val);
  }
  return rational.zero();
}

/**
 * Clones a CPA object with a new quantity.
 */
export function cloneCPAWithQuantity(obj: CPAObject, quantity: Fraction): CPAObject {
  if (obj.kind === "abstracto") {
    return { ...obj, value: quantity };
  }
  if (obj.kind === "forma") {
    return { ...obj, amount: quantity };
  }
  if (obj.kind === "comida") {
    return { ...obj, amount: quantity };
  }
  return obj;
}
