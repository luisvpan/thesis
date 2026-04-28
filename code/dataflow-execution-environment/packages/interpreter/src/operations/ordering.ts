import type Fraction from "fraction.js";
import * as rational from "../runtime/rational";
import type { RuntimeValue, CPAObject } from "../runtime/types";
import {
  isRational,
  isArray,
  isCPAObject,
  getCategoryOrder,
  getTypeKey,
  Category,
} from "../runtime/types";
import { RuntimeError } from "../runtime/errors";

/**
 * Gets the quantity (value for abstract, amount for pictorial/concrete).
 */
function getQuantity(val: RuntimeValue): Fraction {
  if (isRational(val)) {
    return val.value;
  }
  if (isCPAObject(val)) {
    if (val.kind === "abstract") {
      return val.value;
    }
    return val.amount;
  }
  return rational.zero();
}

/**
 * Taxonomical comparison:
 * 1. Category: Concrete (0) < Pictorial (1) < Abstract (2)
 * 2. Type/Subtype: Alphabetical
 * 3. Quantity: Value or Amount
 */
function taxonomicalCompare(a: RuntimeValue, b: RuntimeValue): number {
  // 1. Category ordering
  const catA = getCategoryOrder(a);
  const catB = getCategoryOrder(b);
  if (catA !== catB) {
    return catA - catB;
  }

  // 2. Type/Subtype alphabetical
  const typeA = getTypeKey(a);
  const typeB = getTypeKey(b);
  if (typeA !== typeB) {
    return typeA.localeCompare(typeB);
  }

  // 3. Quantity comparison
  const qtyA = getQuantity(a);
  const qtyB = getQuantity(b);
  return rational.compare(qtyA, qtyB);
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
 * Order ascending operation:
 * - Sort by taxonomical rules (Category > Type > Quantity)
 */
export function orderAsc(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  const sorted = [...flatValues].sort((a, b) => taxonomicalCompare(a, b));

  if (sorted.length === 1) {
    return sorted[0];
  }

  return { kind: "array", elements: sorted };
}

/**
 * Order descending operation:
 * - Sort by taxonomical rules in reverse
 */
export function orderDesc(args: RuntimeValue[]): RuntimeValue {
  const flatValues = flattenArrays(args);

  const sorted = [...flatValues].sort((a, b) => taxonomicalCompare(b, a));

  if (sorted.length === 1) {
    return sorted[0];
  }

  return { kind: "array", elements: sorted };
}
