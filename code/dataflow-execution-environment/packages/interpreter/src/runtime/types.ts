// Runtime value model — LANGUAGE_SPEC.md §1 (Dominio semántico), v0.2.0
//
// Todo valor pertenece a una de tres formas: bolsa, criterio o booleano (§1.1).
// No hay "objeto suelto" ni "arreglo": un objeto individual es una bolsa de una
// entrada, y la bolsa vacía es `nulo` (§1.2.2).

import type Fraction from "fraction.js";
import type { Statement } from "../analyzer/ast";

// =============================================================================
// Identidad CPA (§1.2.1)
// =============================================================================

export type CPACategory = "abstracto" | "pictorico" | "concreto";

/**
 * Una entrada de la bolsa: una identidad CPA (categoría, tipo, subtipo,
 * atributos) asociada a una cantidad racional (§1.2.2).
 */
export interface Entry {
  category: CPACategory;
  type: string;
  subtype: string;
  attributes: Record<string, string>;
  quantity: Fraction;
}

// =============================================================================
// La bolsa (§1.2)
// =============================================================================

/**
 * Secuencia finita y ordenada de entradas. Admite repetidos de la misma
 * identidad (no se agrega por sí sola) y conserva las cantidades 0.
 * Sin entradas es `nulo`: la bolsa vacía, el vector cero (§1.2.5).
 */
export interface Bag {
  kind: "bolsa";
  entries: readonly Entry[];
}

// =============================================================================
// Criterios (§1.3)
// =============================================================================

export type CriterionSubtype = "filter" | "order";
export type OrderDirection = "asc" | "desc";

/**
 * El valor de una propiedad en un criterio: un valor único (una igualdad para
 * el filtro, o una dirección `asc`/`desc` para el orden) o una secuencia de
 * valores que fija el orden explícitamente.
 *
 * La gramática no restringe la forma (§5.1); que la forma corresponda al
 * subtipo es un error estático: criterio inadecuado (§4.2.7).
 */
export type CriterionValue = string | string[];

/**
 * Un criterio declara su subtipo, y ese subtipo determina cómo se interpretan
 * sus valores y qué operación lo consume. Los criterios no se agrupan: cada uno
 * va en su propio `source`.
 */
export interface Criterion {
  kind: "criterio";
  subtype: CriterionSubtype;
  properties: string[];
  values: Record<string, CriterionValue>;
}

// =============================================================================
// Booleano (§1.4)
// =============================================================================

/** Terminal: lo producen las comparaciones y ninguna operación lo consume. */
export interface BooleanValue {
  kind: "booleano";
  value: boolean;
}

export type RuntimeValue = Bag | Criterion | BooleanValue;

/** La categoría de un valor, tal como la usa la pasada estática (§4.2.6). */
export type ValueCategory = "bolsa" | "criterio" | "booleano";

// =============================================================================
// Grafo de ejecución
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
// Type guards
// =============================================================================

export function isBag(val: RuntimeValue): val is Bag {
  return val.kind === "bolsa";
}

export function isCriterion(val: RuntimeValue): val is Criterion {
  return val.kind === "criterio";
}

export function isBoolean(val: RuntimeValue): val is BooleanValue {
  return val.kind === "booleano";
}

export function isFilterCriterion(val: RuntimeValue): val is Criterion {
  return isCriterion(val) && val.subtype === "filter";
}

export function isOrderCriterion(val: RuntimeValue): val is Criterion {
  return isCriterion(val) && val.subtype === "order";
}

export function valueCategory(val: RuntimeValue): ValueCategory {
  return val.kind === "bolsa" ? "bolsa" : val.kind === "criterio" ? "criterio" : "booleano";
}
