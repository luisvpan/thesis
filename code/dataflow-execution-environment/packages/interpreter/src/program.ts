// Consumer-facing Program interface with Fraction types for numeric values
// This is the public API for library consumers to build programs programmatically

import type Fraction from "fraction.js";

// Re-export non-numeric types from AST
export type {
  Operation,
  CategoryValue,
  TypeValue,
  ShapeTypeValue,
  SizeValue,
  FoodTypeValue,
  ColorValue,
} from "./analyzer/ast";

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
  value:
    | import("./analyzer/ast").CategoryValue
    | import("./analyzer/ast").TypeValue
    | import("./analyzer/ast").ShapeTypeValue
    | import("./analyzer/ast").SizeValue
    | import("./analyzer/ast").FoodTypeValue
    | import("./analyzer/ast").ColorValue;
};

// Object Literals with Fraction for numeric fields
export type ObjectLiteral =
  | AbstractObjectLiteral
  | PictorialObjectLiteral
  | ConcreteObjectLiteral
  | SimpleObjectLiteral;

export type SimpleObjectLiteral = {
  type: "ObjectLiteral";
  category?: import("./analyzer/ast").CategoryValue;
  objectType:
    | import("./analyzer/ast").TypeValue
    | import("./analyzer/ast").ShapeTypeValue
    | import("./analyzer/ast").FoodTypeValue;
  subtype?: import("./analyzer/ast").ShapeTypeValue | import("./analyzer/ast").FoodTypeValue;
  size?: import("./analyzer/ast").SizeValue;
  color?: import("./analyzer/ast").ColorValue;
  amount?: Fraction;
  value?: Fraction;
};

export type AbstractObjectLiteral = {
  type: "ObjectLiteral";
  category: "abstracto";
  objectType: "racional";
  value: Fraction;
};

export type PictorialObjectLiteral = {
  type: "ObjectLiteral";
  category: "pictorico";
  objectType: "forma";
  subtype: import("./analyzer/ast").ShapeTypeValue;
  size: import("./analyzer/ast").SizeValue;
  amount: Fraction;
};

export type ConcreteObjectLiteral = {
  type: "ObjectLiteral";
  category: "concreto";
  objectType: "comida";
  subtype: import("./analyzer/ast").FoodTypeValue;
  color: import("./analyzer/ast").ColorValue;
  amount: Fraction;
};
