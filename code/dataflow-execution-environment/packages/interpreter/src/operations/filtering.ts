import type { RuntimeValue } from "../runtime/types";
import { isArray, isOther, isShape, isFood } from "../runtime/types";
import { RuntimeError } from "../runtime/errors";

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
 * Checks if a value matches the criterion.
 * Matches against: size, color, subtype, category, type.
 */
function matchesCriterion(value: RuntimeValue, criterion: string): boolean {
  if (isShape(value)) {
    return (
      value.subtype === criterion ||
      value.size === criterion ||
      value.category === criterion ||
      value.kind === criterion
    );
  }

  if (isFood(value)) {
    return (
      value.subtype === criterion ||
      value.color === criterion ||
      value.category === criterion ||
      value.kind === criterion
    );
  }

  if (isOther(value)) {
    return value.value === criterion;
  }

  // For other types, check if the kind matches
  return value.kind === criterion;
}

/**
 * Filter operation:
 * - First argument(s): items to filter (arrays are flattened)
 * - Last argument: criterion to match
 */
export function filter(args: RuntimeValue[]): RuntimeValue {
  if (args.length < 2) {
    throw new RuntimeError(
      "ARITY_ERROR",
      `filter requires at least 2 arguments, got ${args.length}`
    );
  }

  // Last argument is the criterion
  const criterionArg = args[args.length - 1];
  const items = args.slice(0, -1);

  // Get criterion value
  let criterionValue: string;
  if (isOther(criterionArg)) {
    criterionValue = criterionArg.value;
  } else {
    throw new RuntimeError(
      "TYPE_ERROR",
      "filter criterion must be a keyword value (size, color, type, etc.)"
    );
  }

  // Flatten and filter
  const flatItems = flattenArrays(items);
  const filtered = flatItems.filter((item) =>
    matchesCriterion(item, criterionValue)
  );

  if (filtered.length === 1) {
    return filtered[0];
  }

  return { kind: "array", elements: filtered };
}
