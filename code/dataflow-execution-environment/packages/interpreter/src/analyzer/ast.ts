// AST Node Types for the Dataflow Language v3.0.0

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

// Generic Object Literal with key-value pairs
// All properties are stored as strings (keys without quotes, values as-is)
export type ObjectLiteral = {
  type: "ObjectLiteral";
  properties: ObjectProperty[];
};

export type ObjectProperty = {
  key: string;    // Property name without quotes
  value: string;  // Property value without quotes (or number as string)
};

// Helper type for CPA categories (for type checking)
export type CPACategory = "abstracto" | "pictorico" | "concreto";

// Helper function to get a property value from an ObjectLiteral
export function getProperty(obj: ObjectLiteral, key: string): string | undefined {
  const prop = obj.properties.find(p => p.key === key);
  return prop?.value;
}

// Helper function to check if an ObjectLiteral has all required CPA fields
export function isValidCPAObject(obj: ObjectLiteral): boolean {
  const category = getProperty(obj, "category");
  const type = getProperty(obj, "type");
  const subtype = getProperty(obj, "subtype");
  const quantity = getProperty(obj, "quantity");

  return category !== undefined &&
         type !== undefined &&
         subtype !== undefined &&
         quantity !== undefined;
}
