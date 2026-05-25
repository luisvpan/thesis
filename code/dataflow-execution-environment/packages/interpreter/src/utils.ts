import Fraction from "fraction.js";
import type { CPAObject, CPACategory } from "./runtime/types";

// ============================================================================
// Internal generic creator
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
// Public helpers for each CPA category
// ============================================================================

/**
 * Creates a CPA abstracto representing a rational number.
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
 * Creates a CPA pictórico (visual representation).
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
 * Creates a CPA concreto (physical object).
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
