// Consumer-facing Program interface with Fraction types for numeric values
// This is the public API for library consumers to build programs programmatically

import type Fraction from "fraction.js";

// Re-export non-numeric types from AST
export type { Operation } from "./analyzer/ast";

// Program structure (same as AST)
export type Program = {
  type: "Program";
  statements: Statement[];
};

export type Statement = SourceStatement | TransformStatement | SinkStatement;

export type SourceStatement = {
  type: "SourceStatement";
  identifier: string;
  value: Literal;
};

export type TransformStatement = {
  type: "TransformStatement";
  identifier: string;
  operation: import("./analyzer/ast").Operation;
  arguments: Expression[];
};

export type SinkStatement = {
  type: "SinkStatement";
  identifier: string;
  sourceIdentifier: string;
};

// Expressions
export type Expression = IdentifierExpression | Literal;

export type IdentifierExpression = {
  type: "Identifier";
  name: string;
};

// Literals - v4.0.0: ObjectLiteral = DataLiteral | CriteriaLiteral
export type Literal = ObjectLiteral | OtherLiteral | ArrayLiteral;

export type ArrayLiteral = {
  type: "ArrayLiteral";
  elements: Expression[];
};

export type OtherLiteral = {
  type: "OtherLiteral";
  value: string;
};

// =============================================================================
// Object Literals (v4.0.0) - Discriminated by sourceType
// =============================================================================

// ObjectLiteral es unión de Data y Criteria
export type ObjectLiteral = DataLiteral | CriteriaLiteral;

// Data Literal - CPA objects with category, type, subtype, quantity
export type DataLiteral = {
  type: "DataLiteral";
  sourceType: "data";
  category: string;
  objType: string;      // "type" renamed to avoid keyword conflict
  subtype: string;
  quantity: Fraction;   // Fraction for API convenience
  attributes: Record<string, string>;
};

// Criteria Literal - For filter and order operations
// Generic type: keys of values are constrained to values of properties
export type CriteriaLiteral<P extends string = string> = {
  type: "CriteriaLiteral";
  sourceType: "criteria";
  properties: readonly P[] | P[];  // Permite inferencia de tuplas o as const
  values: Partial<Record<P, string | string[]>>;
};

// =============================================================================
// Type Guards for ObjectLiteral
// =============================================================================

export function isDataLiteral(obj: ObjectLiteral): obj is DataLiteral {
  return obj.type === "DataLiteral";
}

export function isCriteriaLiteral(obj: ObjectLiteral): obj is CriteriaLiteral {
  return obj.type === "CriteriaLiteral";
}

// =============================================================================
// Legacy types (deprecated, for backwards compatibility)
// =============================================================================

/** @deprecated Use DataLiteral or CriteriaLiteral directly */
export type ObjectProperty = {
  key: string;
  value: string | Fraction;
};
