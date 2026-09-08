import type Fraction from "fraction.js";
import type { RuntimeValue, CPAObject, CriteriaObject } from "../runtime/types";
import { isArray, isCPAObject, isCriteria } from "../runtime/types";
import * as rational from "../runtime/rational";

/**
 * Result of separating arguments into data items and criteria elements.
 */
export interface SeparatedArgs {
  dataItems: CPAObject[];
  criteriaElements: (CriteriaObject | CriteriaObject[])[]; // Singles or groups
}

/**
 * Separates arguments into data items and criteria elements.
 * - Data items (CPAObject) are flattened from groups
 * - Criteria elements (CriteriaObject) preserve group structure for AND logic
 */
export function separationPass(args: RuntimeValue[]): SeparatedArgs {
  const dataItems: CPAObject[] = [];
  const criteriaElements: (CriteriaObject | CriteriaObject[])[] = [];

  for (const arg of args) {
    if (isArray(arg)) {
      // Inspect the group
      if (arg.elements.length === 0) continue;

      const first = arg.elements[0];
      if (isCPAObject(first)) {
        // Data group: flatten into dataItems
        for (const el of arg.elements) {
          if (isCPAObject(el)) dataItems.push(el);
        }
      } else if (isCriteria(first)) {
        // Criteria group: preserve as array for AND logic
        const criteriaGroup = arg.elements.filter(isCriteria);
        if (criteriaGroup.length > 0) {
          criteriaElements.push(criteriaGroup);
        }
      }
    } else if (isCPAObject(arg)) {
      dataItems.push(arg);
    } else if (isCriteria(arg)) {
      criteriaElements.push(arg); // Single criteria
    }
  }

  return { dataItems, criteriaElements };
}

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
