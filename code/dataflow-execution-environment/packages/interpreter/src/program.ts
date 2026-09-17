// API pública para construir programas: la misma estructura del AST, pero con
// cantidades ya en `Fraction` y con un único tipo de valor de datos, la bolsa.
//
// Las bolsas se construyen con `createBag()` (ver `bag-builder.ts`); los
// criterios, con `createFilterCriterion` / `createOrderCriterion`.

import type { CriterionSubtype, CriterionValue, Entry } from "./runtime/types";

export type { Operation } from "./analyzer/ast";
export type { CPACategory, CriterionSubtype, CriterionValue, Entry } from "./runtime/types";

type Operation = import("./analyzer/ast").Operation;

// =============================================================================
// Estructura del programa (§2.1)
// =============================================================================

export type Program = {
  type: "Program";
  statements: Statement[];
};

export type Statement = SourceStatement | TransformStatement | SinkStatement;

// Las tres declaraciones tienen su valor opcional: un nodo a medio escribir es
// válido y evalúa a `nulo` (§2.5).

export type SourceStatement = {
  type: "SourceStatement";
  identifier: string;
  value?: Literal;
};

export type TransformStatement = {
  type: "TransformStatement";
  identifier: string;
  /** Una de las operaciones reconocidas (§5.2); otra cosa es un error estático. */
  operation?: Operation | (string & {});
  arguments: Expression[];
};

export type SinkStatement = {
  type: "SinkStatement";
  identifier: string;
  sourceIdentifier?: string;
};

// =============================================================================
// Expresiones — `argument_list ::= identifier ("," identifier)*` (§5.1)
// =============================================================================

export type Expression = IdentifierExpression;

export type IdentifierExpression = {
  type: "Identifier";
  name: string;
};

// =============================================================================
// Literales: una bolsa de datos o un criterio
// =============================================================================

export type Literal = BagLiteral | CriterionLiteral;

/**
 * Una bolsa: 0, 1 o n entradas. Se construye con `createBag()`, que devuelve
 * una bolsa inmutable con `.add()` y `.reset()`.
 */
export type BagLiteral = {
  type: "BagLiteral";
  entries: readonly Entry[];
};

/**
 * Un criterio, con su subtipo declarado (§1.3). Los criterios no se agrupan:
 * cada uno va en su propio `source`.
 */
export type CriterionLiteral<P extends string = string> = {
  type: "CriterionLiteral";
  sourceType: CriterionSubtype;
  properties: readonly P[] | P[];
  values: Partial<Record<P, CriterionValue>>;
};

// =============================================================================
// Type guards
// =============================================================================

export function isBagLiteral(literal: Literal): literal is BagLiteral {
  return literal.type === "BagLiteral";
}

export function isCriterionLiteral(literal: Literal): literal is CriterionLiteral {
  return literal.type === "CriterionLiteral";
}
