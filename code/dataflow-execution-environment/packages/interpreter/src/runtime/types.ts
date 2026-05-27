import type Fraction from "fraction.js";
import type { Statement } from "../analyzer/ast";

// =============================================================================
// CPA Categories
// =============================================================================

export type CPACategory = "abstracto" | "pictorico" | "concreto";

// Category const for taxonomical ordering (lower = higher priority)
export const Category = {
  Concreto: 0,
  Pictorico: 1,
  Abstracto: 2,
} as const;

export type Category = (typeof Category)[keyof typeof Category];

// =============================================================================
// Generic CPA Object - Unified representation for all CPA types
// =============================================================================

export interface GenericCPAObject {
  kind: "cpa";                            // Discriminator for type guards
  category: CPACategory;                  // CPA category
  type: string;                           // Object type: "comida", "forma", "numero", etc.
  subtype: string;                        // Specific subtype: "manzana", "circulo", "racional"
  quantity: Fraction;                     // Unified quantity (amount/value)
  attributes: Record<string, string>;     // Additional key-value attributes
}

// CPAObject is now an alias for the generic type
export type CPAObject = GenericCPAObject;

// =============================================================================
// Criteria Object - For filter and order operations (v4.0.0)
// =============================================================================

export interface CriteriaObject {
  kind: "criteria";
  properties: string[];                      // Properties to evaluate/order by
  values: Record<string, string | string[]>; // Criterion values (can be arrays for sequences)
}

// =============================================================================
// Other Runtime Value Types
// =============================================================================

export type ArrayValue = {
  kind: "arreglo";
  elements: RuntimeValue[];
};

export type OtherValue = {
  kind: "otro";
  value: string;
};

// All possible runtime values
export type RuntimeValue =
  | CPAObject
  | CriteriaObject
  | ArrayValue
  | OtherValue;

// =============================================================================
// Execution Graph Types
// =============================================================================

export type EvaluationState = "pending" | "evaluating" | "completed";

export interface ExecutionNode {
  id: string;
  statement: Statement;
  dependencies: string[];
  dependents: string[];
  state: EvaluationState;
  result?: RuntimeValue;
}

// =============================================================================
// Type Guards
// =============================================================================

export function isArray(val: RuntimeValue): val is ArrayValue {
  return val.kind === "arreglo";
}

export function isCPAObject(val: RuntimeValue): val is CPAObject {
  return val.kind === "cpa";
}

export function isCriteria(val: RuntimeValue): val is CriteriaObject {
  return val.kind === "criteria";
}

export function isOther(val: RuntimeValue): val is OtherValue {
  return val.kind === "otro";
}

/**
 * Check if a criteria object is complete (has values for all its properties)
 */
export function isCriteriaComplete(criteria: CriteriaObject): boolean {
  return criteria.properties.every(prop => prop in criteria.values);
}

// =============================================================================
// CPA Object Helpers
// =============================================================================

/**
 * Get a unique key for CPA aggregation based on category, type, subtype, and attributes
 */
export function getCPAKey(val: CPAObject): string {
  const parts = [val.category, val.type, val.subtype];

  // Include sorted attributes for uniqueness
  const sortedAttrs = Object.entries(val.attributes)
    .sort(([a], [b]) => a.localeCompare(b));

  for (const [key, value] of sortedAttrs) {
    parts.push(`${key}:${value}`);
  }

  return parts.join(":");
}

/**
 * Get category enum value from a runtime value
 */
export function getCategoryOrder(val: RuntimeValue): Category {
  if (isCPAObject(val)) {
    switch (val.category) {
      case "concreto": return Category.Concreto;
      case "pictorico": return Category.Pictorico;
      case "abstracto": return Category.Abstracto;
    }
  }
  return Category.Abstracto;
}

/**
 * Get type key for sorting within categories
 */
export function getTypeKey(val: RuntimeValue): string {
  if (isCPAObject(val)) {
    return `${val.type}:${val.subtype}`;
  }
  return "otro";
}

/**
 * Get quantity from a CPA object
 */
export function getQuantity(val: CPAObject): Fraction {
  return val.quantity;
}

/**
 * Clone a CPA object with a new quantity
 */
export function cloneCPAWithQuantity(obj: CPAObject, quantity: Fraction): CPAObject {
  return { ...obj, quantity };
}

/**
 * Check if a CPA object matches a criterion (for filtering)
 * Matches against category, type, subtype, or any attribute value
 */
export function matchesAttribute(val: CPAObject, criterion: string): boolean {
  if (val.category === criterion) return true;
  if (val.type === criterion) return true;
  if (val.subtype === criterion) return true;
  return Object.values(val.attributes).includes(criterion);
}
