// AST Node Types for the Dataflow Language v4.0.0

export type Program = {
  type: "Program";
  statements: Statement[];
};

export type Statement = SourceStatement | TransformStatement | SinkStatement;

export type SourceStatement = {
  type: "SourceStatement";
  identifier: string;
  value?: Literal;  // Optional for incomplete programs
};

export type TransformStatement = {
  type: "TransformStatement";
  identifier: string;
  operation?: Operation;  // Optional for incomplete programs
  arguments: Expression[];
};

export type SinkStatement = {
  type: "SinkStatement";
  identifier: string;
  sourceIdentifier?: string;  // Optional for incomplete programs
};

// Operations
export type Operation =
  | "sum"
  | "substract"
  | "multiply"
  | "divide"
  | "less_than"
  | "greater_than"
  | "order_asc"
  | "order_desc"
  | "filter";

// Expressions
export type Expression = IdentifierExpression | Literal;

export type IdentifierExpression = {
  type: "Identifier";
  name: string;
};

// Literals
// Note: NumberLiteral is not a top-level literal in grammar v3.1.0
// Numbers are only allowed inside object kvPairs (for quantity values)
export type Literal = ObjectLiteral | StringLiteral | ArrayLiteral | GroupLiteral;

export type StringLiteral = {
  type: "StringLiteral";
  value: string;  // Without quotes
};

export type ArrayLiteral = {
  type: "ArrayLiteral";
  elements: Expression[];
};

// Group Literal - array containing only ObjectLiterals (CPA objects)
export type GroupLiteral = {
  type: "GroupLiteral";
  elements: ObjectLiteral[];
};

// =============================================================================
// Object Literals (v4.0.0) - Discriminated by sourceType
// =============================================================================

// ObjectLiteral is a union of DataLiteral and CriteriaLiteral
export type ObjectLiteral = DataLiteral | CriteriaLiteral;

// Data Literal - CPA objects with category, type, subtype, quantity
export type DataLiteral = {
  type: "DataLiteral";
  sourceType: "data";
  category: string;
  objType: string;      // "type" in source, renamed to avoid keyword
  subtype: string;
  quantity: string;
  attributes: ObjectProperty[];
};

// Criteria Literal - For filter and order operations
export type CriteriaLiteral = {
  type: "CriteriaLiteral";
  sourceType: "criteria";
  properties: string[];       // Properties to match/order by
  values: ObjectProperty[];   // Key-value pairs for criteria values
};

// Key-value pair (value can be string or array of strings)
export type ObjectProperty = {
  key: string;
  value: string | string[];
};

// Helper type for CPA categories (for type checking)
export type CPACategory = "abstracto" | "pictorico" | "concreto";

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
// Helper Functions
// =============================================================================

// Helper function to get a property value from a DataLiteral's attributes
export function getDataAttribute(obj: DataLiteral, key: string): string | string[] | undefined {
  const prop = obj.attributes.find(p => p.key === key);
  return prop?.value;
}

// Helper function to get a value from a CriteriaLiteral
export function getCriteriaValue(obj: CriteriaLiteral, key: string): string | string[] | undefined {
  const prop = obj.values.find(p => p.key === key);
  return prop?.value;
}
