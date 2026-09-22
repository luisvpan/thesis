// AST del lenguaje dataflow — LANGUAGE_SPEC.md §5
//
// Es la salida del parser: la forma textual ya estructurada, con las cantidades
// todavía como texto. La API pública (`program.ts`) es su equivalente con
// `Fraction` y con la bolsa como único valor de datos.

export type { Operation } from "../operations/signatures";

export type Program = {
  type: "Program";
  statements: Statement[];
};

export type Statement = SourceStatement | TransformStatement | SinkStatement;

export type SourceStatement = {
  type: "SourceStatement";
  identifier: string;
  /** Opcional: un `source` sin valor evalúa a `nulo` (§2.5). */
  value?: Literal;
};

export type TransformStatement = {
  type: "TransformStatement";
  identifier: string;
  /**
   * El nombre de la operación, tal cual se escribió: `operation ::= identifier`
   * (§5.1). Que pertenezca al conjunto reconocido lo verifica la pasada
   * estática (§4.2.4). Opcional: un `transform` sin operación evalúa a `nulo`.
   */
  operation?: string;
  arguments: Expression[];
};

export type SinkStatement = {
  type: "SinkStatement";
  identifier: string;
  /** Opcional: un `sink` sin fuente evalúa a `nulo` (§2.5). */
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
// Literales
// =============================================================================

export type Literal = ObjectLiteral | GroupLiteral;

/** `object_literal ::= data_literal | criteria_literal` */
export type ObjectLiteral = DataLiteral | CriterionLiteral;

/** `group ::= "[" (data_literal ("," data_literal)*)? "]"` — solo datos (§4.1). */
export type GroupLiteral = {
  type: "GroupLiteral";
  elements: DataLiteral[];
};

export type DataLiteral = {
  type: "DataLiteral";
  sourceType: "data";
  category: string;
  /** `"type"` en el texto; renombrado para no chocar con el discriminante. */
  objType: string;
  subtype: string;
  quantity: string;
  attributes: ObjectProperty[];
};

/** `criteria_kind ::= '"filter"' | '"order"'` (§5.1). */
export type CriterionSubtype = "filter" | "order";

export type CriterionLiteral = {
  type: "CriterionLiteral";
  sourceType: CriterionSubtype;
  properties: string[];
  values: ObjectProperty[];
};

export type ObjectProperty = {
  key: string;
  value: string | string[];
};

export type CPACategory = "abstracto" | "pictorico" | "concreto";

export const CPA_CATEGORIES: readonly CPACategory[] = ["abstracto", "pictorico", "concreto"];

export function isCPACategory(value: string): value is CPACategory {
  return (CPA_CATEGORIES as readonly string[]).includes(value);
}

// =============================================================================
// Type guards
// =============================================================================

export function isDataLiteral(literal: Literal): literal is DataLiteral {
  return literal.type === "DataLiteral";
}

export function isCriterionLiteral(literal: Literal): literal is CriterionLiteral {
  return literal.type === "CriterionLiteral";
}

export function isGroupLiteral(literal: Literal): literal is GroupLiteral {
  return literal.type === "GroupLiteral";
}

export function getDataAttribute(obj: DataLiteral, key: string): string | string[] | undefined {
  return obj.attributes.find((property) => property.key === key)?.value;
}

export function getCriterionValue(obj: CriterionLiteral, key: string): string | string[] | undefined {
  return obj.values.find((property) => property.key === key)?.value;
}
