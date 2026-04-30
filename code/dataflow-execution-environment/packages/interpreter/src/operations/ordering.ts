import * as rational from "../runtime/rational";
import type { RuntimeValue } from "../runtime/types";
import {
  getCategoryOrder,
  getTypeKey,
} from "../runtime/types";
import { flattenArrays, getQuantityOrZero } from "./utils";

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
  const qtyA = getQuantityOrZero(a);
  const qtyB = getQuantityOrZero(b);
  return rational.compare(qtyA, qtyB);
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

  return { kind: "arreglo", elements: sorted };
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

  return { kind: "arreglo", elements: sorted };
}
