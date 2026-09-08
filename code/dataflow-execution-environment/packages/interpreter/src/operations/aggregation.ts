import type { RuntimeValue } from "../runtime/types";
import { isCPAObject } from "../runtime/types";
import * as rational from "../runtime/rational";
import { flattenArrays, getQuantity } from "./utils";
import { createAbstractNumber } from "../utils";

/**
 * Counts the total quantity of all elements.
 * - Flattens all input arrays
 * - Sums the quantities of all CPA objects
 * - Returns an abstract number with the total
 */
export function count(args: RuntimeValue[]): RuntimeValue {
  const flat = flattenArrays(args);

  let total = rational.zero();

  for (const val of flat) {
    if (isCPAObject(val)) {
      total = rational.add(total, getQuantity(val));
    }
  }

  return createAbstractNumber(total);
}
