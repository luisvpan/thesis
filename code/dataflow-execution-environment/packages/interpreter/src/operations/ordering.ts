import * as rational from "../runtime/rational";
import type { RuntimeValue, CPAObject, CriteriaObject } from "../runtime/types";
import {
  getCategoryOrder,
  getTypeKey,
  isCriteriaComplete,
} from "../runtime/types";
import { separationPass, getQuantityOrZero } from "./utils";

/**
 * Taxonomical comparison (fallback when no criteria):
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
 * Get a property value from a CPAObject for comparison.
 * Supports: category, type, subtype, quantity, or any attribute key.
 */
function getItemProperty(item: CPAObject, prop: string): string {
  switch (prop) {
    case "category":
      return item.category;
    case "type":
      return item.type;
    case "subtype":
      return item.subtype;
    case "quantity":
      return item.quantity.valueOf().toString();
    default:
      return item.attributes[prop] ?? "";
  }
}

/**
 * Compare two items by a property with optional sequence values.
 * If sequenceValues is an array, items are ordered by their position in the sequence.
 * Otherwise, alphabetical comparison is used.
 */
function compareByProperty(
  a: CPAObject,
  b: CPAObject,
  prop: string,
  sequenceValues: string | string[] | undefined,
  isAscending: boolean
): number {
  const valA = getItemProperty(a, prop);
  const valB = getItemProperty(b, prop);

  let cmp: number;

  if (Array.isArray(sequenceValues)) {
    // Order by position in sequence (items not in sequence go to the end)
    const indexA = sequenceValues.indexOf(valA);
    const indexB = sequenceValues.indexOf(valB);
    const posA = indexA === -1 ? Infinity : indexA;
    const posB = indexB === -1 ? Infinity : indexB;
    cmp = posA - posB;
  } else {
    // Alphabetical comparison
    cmp = valA.localeCompare(valB);
  }

  return isAscending ? cmp : -cmp;
}

/**
 * Compiled criteria from RTL compilation.
 */
interface CompiledCriteria {
  properties: string[];
  values: Record<string, string | string[]>;
}

/**
 * Compile criteria right-to-left.
 * Later criteria have priority - their properties come first in the sort order,
 * and their values override earlier values for the same property.
 */
function compileCriteriaRTL(criteriaElements: (CriteriaObject | CriteriaObject[])[]): CompiledCriteria {
  // 1. Unroll criteria groups into flat timeline
  const timeline: CriteriaObject[] = [];
  for (const element of criteriaElements) {
    if (Array.isArray(element)) {
      timeline.push(...element);
    } else {
      timeline.push(element);
    }
  }

  // 2. Filter out incomplete criteria (those without values for their properties)
  const completeCriteria = timeline.filter(isCriteriaComplete);

  // 3. Compile right-to-left
  const finalProperties: string[] = [];
  const finalValues: Record<string, string | string[]> = {};

  for (let i = completeCriteria.length - 1; i >= 0; i--) {
    const current = completeCriteria[i];

    // Add properties in order (RTL means later ones come first)
    for (const prop of current.properties) {
      if (!finalProperties.includes(prop)) {
        finalProperties.push(prop);
      }
    }

    // Merge values (RTL means later values have priority)
    for (const [key, value] of Object.entries(current.values)) {
      if (!(key in finalValues)) {
        finalValues[key] = value;
      }
    }
  }

  return { properties: finalProperties, values: finalValues };
}

/**
 * Execute ordering with criteria-based sorting.
 */
function executeOrder(args: RuntimeValue[], isAscending: boolean): RuntimeValue {
  const { dataItems, criteriaElements } = separationPass(args);

  // If no data items, return empty array
  if (dataItems.length === 0) {
    return { kind: "arreglo", elements: [] };
  }

  // If no criteria, use taxonomical ordering as fallback
  if (criteriaElements.length === 0) {
    const sorted = [...dataItems].sort((a, b) => {
      const cmp = taxonomicalCompare(a, b);
      return isAscending ? cmp : -cmp;
    });

    if (sorted.length === 1) {
      return sorted[0];
    }
    return { kind: "arreglo", elements: sorted };
  }

  // Compile criteria RTL
  const compiled = compileCriteriaRTL(criteriaElements);

  // If no complete criteria after compilation, use taxonomical fallback
  if (compiled.properties.length === 0) {
    const sorted = [...dataItems].sort((a, b) => {
      const cmp = taxonomicalCompare(a, b);
      return isAscending ? cmp : -cmp;
    });

    if (sorted.length === 1) {
      return sorted[0];
    }
    return { kind: "arreglo", elements: sorted };
  }

  // Sort by compiled criteria
  const sorted = [...dataItems].sort((a, b) => {
    for (const prop of compiled.properties) {
      const cmp = compareByProperty(a, b, prop, compiled.values[prop], isAscending);
      if (cmp !== 0) return cmp;
    }
    // Fallback to taxonomical if all criteria equal
    const cmp = taxonomicalCompare(a, b);
    return isAscending ? cmp : -cmp;
  });

  if (sorted.length === 1) {
    return sorted[0];
  }

  return { kind: "arreglo", elements: sorted };
}

/**
 * Order ascending operation:
 * - Uses criteria-based sorting with RTL compilation
 * - Falls back to taxonomical rules if no criteria
 */
export function orderAsc(args: RuntimeValue[]): RuntimeValue {
  return executeOrder(args, true);
}

/**
 * Order descending operation:
 * - Uses criteria-based sorting with RTL compilation (reversed)
 * - Falls back to taxonomical rules if no criteria
 */
export function orderDesc(args: RuntimeValue[]): RuntimeValue {
  return executeOrder(args, false);
}
