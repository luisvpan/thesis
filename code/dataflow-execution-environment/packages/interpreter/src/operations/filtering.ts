import type { RuntimeValue, CPAObject, CriteriaObject } from "../runtime/types";
import { isOther, isCPAObject, matchesAttribute, isCriteriaComplete } from "../runtime/types";
import { RuntimeError } from "../runtime/errors";
import { separationPass } from "./utils";

/**
 * Gets a property value from a CPA object by property name.
 * Checks category, type, subtype, and attributes.
 */
function getItemProperty(item: CPAObject, prop: string): string | undefined {
  if (prop === "category") return item.category;
  if (prop === "type") return item.type;
  if (prop === "subtype") return item.subtype;
  return item.attributes[prop];
}

/**
 * Checks if a CPA object matches a single CriteriaObject.
 * All properties specified in the criterion must match.
 */
function matchItem(item: CPAObject, criterion: CriteriaObject): boolean {
  // Incomplete criteria (no values for properties) don't filter anything
  if (!isCriteriaComplete(criterion)) {
    return true;
  }

  // All properties must match
  for (const prop of criterion.properties) {
    const criterionValue = criterion.values[prop];
    const itemValue = getItemProperty(item, prop);

    if (itemValue === undefined) {
      return false;
    }

    if (Array.isArray(criterionValue)) {
      // Criterion has multiple acceptable values (OR within property)
      if (!criterionValue.includes(itemValue)) return false;
    } else {
      // Single value match
      if (itemValue !== criterionValue) return false;
    }
  }
  return true;
}

/**
 * Legacy: Checks if a value matches a string criterion.
 * For backward compatibility with old filter(items, "keyword") syntax.
 */
function matchesCriterion(value: RuntimeValue, criterion: string): boolean {
  if (isCPAObject(value)) {
    return matchesAttribute(value, criterion);
  }

  if (isOther(value)) {
    return value.value === criterion;
  }

  return value.kind === criterion;
}

/**
 * Filter operation (v4.0.0):
 * - Uses separationPass to separate data items from criteria
 * - OR of ANDs logic: groups of criteria = AND, separate criteria = OR
 * - Incomplete criteria are ignored (don't filter anything)
 *
 * Examples:
 * - filter(data, c1, c2) → items matching c1 OR c2
 * - filter(data, [c1, c2]) → items matching (c1 AND c2)
 * - filter(data, [c1, c2], c3) → items matching (c1 AND c2) OR c3
 */
export function filter(args: RuntimeValue[]): RuntimeValue {
  if (args.length < 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `filter requires at least 2 arguments, got ${args.length}`
    );
  }

  const { dataItems, criteriaElements } = separationPass(args);

  // Legacy support: if no criteria objects found, check for string criterion
  if (criteriaElements.length === 0) {
    const lastArg = args[args.length - 1];
    if (isOther(lastArg)) {
      const criterionValue = lastArg.value;
      const filtered = dataItems.filter((item) =>
        matchesCriterion(item, criterionValue)
      );
      if (filtered.length === 1) return filtered[0];
      return { kind: "arreglo", elements: filtered };
    }
  }

  // No criteria to filter by - return all data items
  if (criteriaElements.length === 0) {
    if (dataItems.length === 1) return dataItems[0];
    return { kind: "arreglo", elements: dataItems };
  }

  // Filter with OR of ANDs logic
  const filtered = dataItems.filter((item) => {
    // Top-level: OR between branches
    return criteriaElements.some((branch) => {
      if (Array.isArray(branch)) {
        // Criteria group: AND of all criteria in the group
        return branch.every((criterion) => matchItem(item, criterion));
      } else {
        // Single criteria
        return matchItem(item, branch);
      }
    });
  });

  if (filtered.length === 1) return filtered[0];
  return { kind: "arreglo", elements: filtered };
}
