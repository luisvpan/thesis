// AST Node Types for the Dataflow Language v2.1.0

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
  operation: Operation;
  arguments: Expression[];
};

export type SinkStatement = {
  type: "SinkStatement";
  identifier: string;
  sourceIdentifier: string;
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
export type Literal = ObjectLiteral | OtherLiteral | ArrayLiteral | NumberLiteral;

export type NumberLiteral = {
  type: "NumberLiteral";
  value: number;
};

export type ArrayLiteral = {
  type: "ArrayLiteral";
  elements: Expression[];
};

export type OtherLiteral = {
  type: "OtherLiteral";
  value: CategoryValue | TypeValue | ShapeTypeValue | SizeValue | FoodTypeValue | ColorValue;
};

// Object Literals with category-specific extensions
export type ObjectLiteral = AbstractObjectLiteral | PictorialObjectLiteral | ConcreteObjectLiteral | SimpleObjectLiteral;

export type SimpleObjectLiteral = {
  type: "ObjectLiteral";
  category?: CategoryValue;
  objectType: TypeValue | ShapeTypeValue | FoodTypeValue;
  subtype?: ShapeTypeValue | FoodTypeValue;
  size?: SizeValue;
  color?: ColorValue;
  amount?: number;
  value?: number;
};

export type AbstractObjectLiteral = {
  type: "ObjectLiteral";
  category: "abstract";
  objectType: "rational";
  value: number;
};

export type PictorialObjectLiteral = {
  type: "ObjectLiteral";
  category: "pictorial";
  objectType: "shape";
  subtype: ShapeTypeValue;
  size: SizeValue;
  amount: number;
};

export type ConcreteObjectLiteral = {
  type: "ObjectLiteral";
  category: "concrete";
  objectType: "food";
  subtype: FoodTypeValue;
  color: ColorValue;
  amount: number;
};

// Value types (v2.1.0: rational instead of integer)
export type CategoryValue = "abstract" | "pictorial" | "concrete";

export type TypeValue = "rational" | "shape" | "food";

export type ShapeTypeValue = "circle" | "square";

export type SizeValue = "small" | "medium" | "large";

export type FoodTypeValue = "grape" | "pear" | "apple" | "burger";

export type ColorValue = "purple" | "green" | "red" | "orange";
