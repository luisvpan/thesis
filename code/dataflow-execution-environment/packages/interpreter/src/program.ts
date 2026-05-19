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

// Literals - NumberLiteral uses Fraction instead of string
export type Literal = ObjectLiteral | OtherLiteral | ArrayLiteral | NumberLiteral;

export type NumberLiteral = {
  type: "NumberLiteral";
  value: Fraction;
};

export type ArrayLiteral = {
  type: "ArrayLiteral";
  elements: Expression[];
};

export type OtherLiteral = {
  type: "OtherLiteral";
  value: string;
};

// Object property with Fraction for numeric values
export type ObjectProperty = {
  key: string;
  value: string | Fraction;
};

// Generic Object Literal with key-value pairs
// Numeric values (like quantity) are stored as Fraction
export type ObjectLiteral = {
  type: "ObjectLiteral";
  properties: ObjectProperty[];
  // Convenience accessor for quantity (computed from properties)
  quantity?: Fraction;
};
