import type Fraction from "fraction.js";
import type { RuntimeValue, BooleanValue, CPAObject } from "../runtime/types";
import { isArray, isCPAObject, getCPAKey } from "../runtime/types";
import * as rational from "../runtime/rational";
import { RuntimeError } from "../runtime/errors";
import { flattenArrays, getQuantity } from "./utils";

/**
 * Normalizes a RuntimeValue to a canonical form: a Map of CPAKey -> Fraction.
 * This allows comparing arrays with single elements that have equivalent quantities.
 *
 * Examples:
 * - [apple, apple, apple] -> { "concreto:food:apple": 3 }
 * - { ...apple, qty: 3 }  -> { "concreto:food:apple": 3 }
 */
function normalize(val: RuntimeValue): Map<string, Fraction> {
  const result = new Map<string, Fraction>();

  const flat = isArray(val) ? flattenArrays(val.elements) : [val];

  for (const item of flat) {
    if (isCPAObject(item)) {
      const key = getCPAKey(item);
      const existing = result.get(key) ?? rational.zero();
      result.set(key, rational.add(existing, getQuantity(item)));
    }
  }

  return result;
}

/**
 * Compares two normalized maps for deep equality.
 * Two maps are equal if they have the same keys and each key maps to the same quantity.
 */
function mapsAreEqual(a: Map<string, Fraction>, b: Map<string, Fraction>): boolean {
  if (a.size !== b.size) return false;

  for (const [key, qtyA] of a) {
    const qtyB = b.get(key);
    if (qtyB === undefined) return false;
    if (!qtyA.equals(qtyB)) return false;
  }

  return true;
}

/**
 * Compare operation: checks if two values are semantically equal.
 *
 * - Normalizes both arguments to canonical form (grouped by CPAKey with summed quantities)
 * - Returns true if the normalized forms are equal
 *
 * Special behavior:
 * - compare([apple, apple, apple], { apple, qty: 3 }) -> true
 * - compare([apple, apple], [apple, apple, apple]) -> false
 * - compare([apple, apple, pear], [2 apples, pear]) -> true
 */
export function compare(args: RuntimeValue[]): BooleanValue {
  if (args.length !== 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `compare requires exactly 2 arguments, got ${args.length}`
    );
  }

  const [a, b] = args;

  const normalizedA = normalize(a);
  const normalizedB = normalize(b);

  const equal = mapsAreEqual(normalizedA, normalizedB);

  return { kind: "booleano", value: equal };
}
