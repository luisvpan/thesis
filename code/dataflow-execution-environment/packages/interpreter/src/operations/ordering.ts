import type { RuntimeValue, CPAObject, CriteriaObject } from "../runtime/types";
import { isCriteriaComplete } from "../runtime/types";
import { separationPass } from "./utils";

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
 * Wrap result: single element returns unwrapped, multiple as array.
 */
function wrapResult(items: CPAObject[]): RuntimeValue {
  if (items.length === 1) {
    return items[0];
  }
  return { kind: "arreglo", elements: items };
}

/**
 * Execute ordering with criteria-based sorting.
 *
 * Behavior:
 * - No criteria: returns data in original order (no-op)
 * - With criteria: stable sort by criteria, ties maintain original order
 */
function executeOrder(args: RuntimeValue[], isAscending: boolean): RuntimeValue {
  const { dataItems, criteriaElements } = separationPass(args);

  // If no data items, return empty array
  if (dataItems.length === 0) {
    return { kind: "arreglo", elements: [] };
  }

  // No criteria = no-op (return in original order)
  if (criteriaElements.length === 0) {
    return wrapResult(dataItems);
  }

  // Compile criteria RTL
  const compiled = compileCriteriaRTL(criteriaElements);

  // No complete criteria after compilation = no-op
  if (compiled.properties.length === 0) {
    return wrapResult(dataItems);
  }

  // Stable sort: use original index as tiebreaker
  const indexed = dataItems.map((item, i) => ({ item, i }));
  indexed.sort((a, b) => {
    for (const prop of compiled.properties) {
      const cmp = compareByProperty(a.item, b.item, prop, compiled.values[prop], isAscending);
      if (cmp !== 0) return cmp;
    }
    // Tie: maintain original order
    return a.i - b.i;
  });

  const sorted = indexed.map(x => x.item);
  return wrapResult(sorted);
}

/**
 * Order ascending operation:
 * - No criteria: returns data in original order (no-op)
 * - With criteria: stable sort ascending by criteria properties
 */
export function orderAsc(args: RuntimeValue[]): RuntimeValue {
  return executeOrder(args, true);
}

/**
 * Order descending operation:
 * - No criteria: returns data in original order (no-op)
 * - With criteria: stable sort descending by criteria properties
 */
export function orderDesc(args: RuntimeValue[]): RuntimeValue {
  return executeOrder(args, false);
}
