import Fraction from "fraction.js";
import type { CPAObject, CPACategory } from "./runtime/types";
import type { DataLiteral, CriteriaLiteral } from "./program";

// ============================================================================
// Runtime object helpers (for internal use)
// ============================================================================

interface CreateObjectParams {
  category: CPACategory;
  type: string;
  subtype: string;
  quantity: number | string | Fraction;
  attributes?: Record<string, string>;
}

const createObject = ({
  category,
  type,
  subtype,
  quantity,
  attributes = {},
}: CreateObjectParams) =>
  ({
    kind: "cpa" as const,
    category,
    type,
    subtype,
    quantity: quantity instanceof Fraction ? quantity : new Fraction(quantity),
    attributes,
  }) satisfies CPAObject;

// ============================================================================
// Runtime helpers for each CPA category
// ============================================================================

/**
 * Creates a runtime CPA abstracto representing a rational number.
 */
export const createAbstractNumber = (
  quantity: number | string | Fraction,
  attributes?: Record<string, string>
) =>
  createObject({
    category: "abstracto",
    type: "numero",
    subtype: "racional",
    quantity,
    attributes,
  }) satisfies CPAObject;

/**
 * Creates a runtime CPA pictórico (visual representation).
 */
export const createPictoricObject = (
  type: string,
  subtype: string,
  quantity: number | string | Fraction,
  attributes?: Record<string, string>
) =>
  createObject({
    category: "pictorico",
    type,
    subtype,
    quantity,
    attributes,
  }) satisfies CPAObject;

/**
 * Creates a runtime CPA concreto (physical object).
 */
export const createConcreteObject = (
  type: string,
  subtype: string,
  quantity: number | string | Fraction,
  attributes?: Record<string, string>
) =>
  createObject({
    category: "concreto",
    type,
    subtype,
    quantity,
    attributes,
  }) satisfies CPAObject;

// ============================================================================
// AST Literal helpers (for building programs programmatically)
// ============================================================================

/**
 * Creates an AST DataLiteral for an abstract number.
 */
export const createAbstractDataLiteral = (
  quantity: number | Fraction,
  attributes?: Record<string, string>
): DataLiteral => ({
  type: "DataLiteral",
  sourceType: "data",
  category: "abstracto",
  objType: "numero",
  subtype: "racional",
  quantity: quantity instanceof Fraction ? quantity : new Fraction(quantity),
  attributes: attributes ?? {},
});

/**
 * Creates an AST DataLiteral for a pictórico (visual representation).
 */
export const createPictoricDataLiteral = (
  objType: string,
  subtype: string,
  quantity: number | Fraction,
  attributes?: Record<string, string>
): DataLiteral => ({
  type: "DataLiteral",
  sourceType: "data",
  category: "pictorico",
  objType,
  subtype,
  quantity: quantity instanceof Fraction ? quantity : new Fraction(quantity),
  attributes: attributes ?? {},
});

/**
 * Creates an AST DataLiteral for a concreto (physical object).
 */
export const createConcreteDataLiteral = (
  objType: string,
  subtype: string,
  quantity: number | Fraction,
  attributes?: Record<string, string>
): DataLiteral => ({
  type: "DataLiteral",
  sourceType: "data",
  category: "concreto",
  objType,
  subtype,
  quantity: quantity instanceof Fraction ? quantity : new Fraction(quantity),
  attributes: attributes ?? {},
});

/**
 * Creates an AST CriteriaLiteral for filter/order operations.
 * Uses const type parameter for tuple inference.
 *
 * @example
 * const criteria = createCriteriaLiteral({
 *   properties: ["color", "size"],  // Infiere P = "color" | "size"
 *   values: { color: "rojo" }       // Solo acepta "color" | "size" como keys
 * });
 */
export const createCriteriaLiteral = <const P extends string>(
  criteria: Omit<CriteriaLiteral<P>, "type" | "sourceType">
): CriteriaLiteral<P> => ({
  type: "CriteriaLiteral",
  sourceType: "criteria",
  ...criteria,
});
