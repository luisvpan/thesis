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
  value: string;
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
  amount?: string;
  value?: string;
};

export type AbstractObjectLiteral = {
  type: "ObjectLiteral";
  category: "abstracto";
  objectType: "racional";
  value: string;
};

export type PictorialObjectLiteral = {
  type: "ObjectLiteral";
  category: "pictorico";
  objectType: "forma";
  subtype: ShapeTypeValue;
  size: SizeValue;
  amount: string;
};

export type ConcreteObjectLiteral = {
  type: "ObjectLiteral";
  category: "concreto";
  objectType: "comida";
  subtype: FoodTypeValue;
  color: ColorValue;
  amount: string;
};

// Value types (v2.1.0 - Spanish)
export type CategoryValue = "abstracto" | "pictorico" | "concreto";

export type TypeValue = "racional" | "forma" | "comida";

export type ShapeTypeValue = "circulo" | "cuadrado" | "triangulo" | "rectangulo" | "rombo" | "estrella" | "trapecio";

export type SizeValue = "pequeño" | "mediano" | "grande";

export type FoodTypeValue = "manzana" | "hamburguesa" | "uva" | "pasta" | "pera";

export type ColorValue = "morado" | "verde" | "rojo" | "naranja" | "azul" | "amarillo";
