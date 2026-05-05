import { CstParser } from "chevrotain";
import {
  allTokens,
  Source,
  Transform,
  Sink,
  Abstract,
  Pictorial,
  Concrete,
  RationalType,
  ShapeType,
  FoodType,
  MontessoriType,
  Circle,
  Square,
  Triangle,
  Rectangle,
  Diamond,
  Star,
  Trapezoid,
  Grape,
  Pear,
  Apple,
  Burger,
  Pasta,
  Small,
  Medium,
  Large,
  Purple,
  Green,
  Red,
  Orange,
  Blue,
  Yellow,
  Sum,
  Substract,
  Multiply,
  Divide,
  LessThan,
  GreaterThan,
  OrderAsc,
  OrderDesc,
  Filter,
  Category,
  Type,
  Value,
  Subtype,
  Size,
  Amount,
  Color,
  Equals,
  Semicolon,
  Comma,
  Colon,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  NumberLiteral,
  Identifier,
} from "./lexer";

export class DataflowParser extends CstParser {
  constructor() {
    super(allTokens);
    this.performSelfAnalysis();
  }

  // program ::= statement*
  public program = this.RULE("program", () => {
    this.MANY(() => {
      this.SUBRULE(this.statement);
    });
  });

  // statement ::= source_statement | transform_statement | sink_statement
  private statement = this.RULE("statement", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.sourceStatement) },
      { ALT: () => this.SUBRULE(this.transformStatement) },
      { ALT: () => this.SUBRULE(this.sinkStatement) },
    ]);
  });

  // source_statement ::= "source" identifier "=" literal ";"
  private sourceStatement = this.RULE("sourceStatement", () => {
    this.CONSUME(Source);
    this.CONSUME(Identifier);
    this.CONSUME(Equals);
    this.SUBRULE(this.literal);
    this.CONSUME(Semicolon);
  });

  // transform_statement ::= "transform" identifier "=" operation "(" argument_list ")" ";"
  private transformStatement = this.RULE("transformStatement", () => {
    this.CONSUME(Transform);
    this.CONSUME(Identifier);
    this.CONSUME(Equals);
    this.SUBRULE(this.operation);
    this.CONSUME(LParen);
    this.SUBRULE(this.argumentList);
    this.CONSUME(RParen);
    this.CONSUME(Semicolon);
  });

  // sink_statement ::= "sink" identifier "=" identifier ";"
  private sinkStatement = this.RULE("sinkStatement", () => {
    this.CONSUME(Sink);
    this.CONSUME1(Identifier);
    this.CONSUME(Equals);
    this.CONSUME2(Identifier);
    this.CONSUME(Semicolon);
  });

  // operation ::= "sum" | "substract" | "multiply" | "divide" | "less_than" | "greater_than" | "order_asc" | "order_desc" | "filter"
  private operation = this.RULE("operation", () => {
    this.OR([
      { ALT: () => this.CONSUME(Sum) },
      { ALT: () => this.CONSUME(Substract) },
      { ALT: () => this.CONSUME(Multiply) },
      { ALT: () => this.CONSUME(Divide) },
      { ALT: () => this.CONSUME(LessThan) },
      { ALT: () => this.CONSUME(GreaterThan) },
      { ALT: () => this.CONSUME(OrderAsc) },
      { ALT: () => this.CONSUME(OrderDesc) },
      { ALT: () => this.CONSUME(Filter) },
    ]);
  });

  // argument_list ::= expression ("," expression)*
  private argumentList = this.RULE("argumentList", () => {
    this.SUBRULE(this.expression);
    this.MANY(() => {
      this.CONSUME(Comma);
      this.SUBRULE2(this.expression);
    });
  });

  // expression ::= identifier | literal
  private expression = this.RULE("expression", () => {
    this.OR([
      { ALT: () => this.CONSUME(Identifier) },
      { ALT: () => this.SUBRULE(this.literal) },
    ]);
  });

  // literal ::= object_literal | other_literal | array_literal | number_literal
  private literal = this.RULE("literal", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.objectLiteral) },
      { ALT: () => this.SUBRULE(this.arrayLiteral) },
      { ALT: () => this.SUBRULE(this.numberLiteral) },
      { ALT: () => this.SUBRULE(this.otherLiteral) },
    ]);
  });

  // number_literal ::= rational (sign now included in token pattern)
  private numberLiteral = this.RULE("numberLiteral", () => {
    this.CONSUME(NumberLiteral);
  });

  // array_literal ::= "[" expression ("," expression)* "]"
  private arrayLiteral = this.RULE("arrayLiteral", () => {
    this.CONSUME(LBracket);
    this.SUBRULE(this.expression);
    this.MANY(() => {
      this.CONSUME(Comma);
      this.SUBRULE2(this.expression);
    });
    this.CONSUME(RBracket);
  });

  // object_literal ::= "{" object_properties "}"
  private objectLiteral = this.RULE("objectLiteral", () => {
    this.CONSUME(LBrace);
    this.SUBRULE(this.objectProperties);
    this.CONSUME(RBrace);
  });

  // Object properties - flexible parsing for various object formats
  private objectProperties = this.RULE("objectProperties", () => {
    this.SUBRULE(this.objectProperty);
    this.MANY(() => {
      this.CONSUME(Comma);
      this.SUBRULE2(this.objectProperty);
    });
  });

  // Single object property
  private objectProperty = this.RULE("objectProperty", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.categoryProperty) },
      { ALT: () => this.SUBRULE(this.typeProperty) },
      { ALT: () => this.SUBRULE(this.valueProperty) },
      { ALT: () => this.SUBRULE(this.subtypeProperty) },
      { ALT: () => this.SUBRULE(this.sizeProperty) },
      { ALT: () => this.SUBRULE(this.amountProperty) },
      { ALT: () => this.SUBRULE(this.colorProperty) },
    ]);
  });

  // category : categoryValue
  private categoryProperty = this.RULE("categoryProperty", () => {
    this.CONSUME(Category);
    this.CONSUME(Colon);
    this.SUBRULE(this.categoryValue);
  });

  // type : typeValue
  private typeProperty = this.RULE("typeProperty", () => {
    this.CONSUME(Type);
    this.CONSUME(Colon);
    this.SUBRULE(this.typeValue);
  });

  // value : number
  private valueProperty = this.RULE("valueProperty", () => {
    this.CONSUME(Value);
    this.CONSUME(Colon);
    this.SUBRULE(this.numberLiteral);
  });

  // subtype : subtypeValue
  private subtypeProperty = this.RULE("subtypeProperty", () => {
    this.CONSUME(Subtype);
    this.CONSUME(Colon);
    this.SUBRULE(this.subtypeValue);
  });

  // size : sizeValue
  private sizeProperty = this.RULE("sizeProperty", () => {
    this.CONSUME(Size);
    this.CONSUME(Colon);
    this.SUBRULE(this.sizeValue);
  });

  // amount : number
  private amountProperty = this.RULE("amountProperty", () => {
    this.CONSUME(Amount);
    this.CONSUME(Colon);
    this.SUBRULE(this.numberLiteral);
  });

  // color : colorValue
  private colorProperty = this.RULE("colorProperty", () => {
    this.CONSUME(Color);
    this.CONSUME(Colon);
    this.SUBRULE(this.colorValue);
  });

  // category ::= "abstract" | "pictorial" | "concrete"
  private categoryValue = this.RULE("categoryValue", () => {
    this.OR([
      { ALT: () => this.CONSUME(Abstract) },
      { ALT: () => this.CONSUME(Pictorial) },
      { ALT: () => this.CONSUME(Concrete) },
    ]);
  });

  // typeValue - all type-like values including subtypes for flexibility
  private typeValue = this.RULE("typeValue", () => {
    this.OR([
      { ALT: () => this.CONSUME(RationalType) },
      { ALT: () => this.CONSUME(ShapeType) },
      { ALT: () => this.CONSUME(FoodType) },
      { ALT: () => this.CONSUME(MontessoriType) },
      // Shape subtypes
      { ALT: () => this.CONSUME(Circle) },
      { ALT: () => this.CONSUME(Square) },
      { ALT: () => this.CONSUME(Triangle) },
      { ALT: () => this.CONSUME(Rectangle) },
      { ALT: () => this.CONSUME(Diamond) },
      { ALT: () => this.CONSUME(Star) },
      { ALT: () => this.CONSUME(Trapezoid) },
      // Food subtypes
      { ALT: () => this.CONSUME(Grape) },
      { ALT: () => this.CONSUME(Pear) },
      { ALT: () => this.CONSUME(Apple) },
      { ALT: () => this.CONSUME(Burger) },
      { ALT: () => this.CONSUME(Pasta) },
    ]);
  });

  // subtypeValue ::= shapeType | foodType
  private subtypeValue = this.RULE("subtypeValue", () => {
    this.OR([
      // Shape subtypes
      { ALT: () => this.CONSUME(Circle) },
      { ALT: () => this.CONSUME(Square) },
      { ALT: () => this.CONSUME(Triangle) },
      { ALT: () => this.CONSUME(Rectangle) },
      { ALT: () => this.CONSUME(Diamond) },
      { ALT: () => this.CONSUME(Star) },
      { ALT: () => this.CONSUME(Trapezoid) },
      // Food subtypes
      { ALT: () => this.CONSUME(Grape) },
      { ALT: () => this.CONSUME(Pear) },
      { ALT: () => this.CONSUME(Apple) },
      { ALT: () => this.CONSUME(Burger) },
      { ALT: () => this.CONSUME(Pasta) },
    ]);
  });

  // size_value ::= "pequeño" | "mediano" | "grande"
  private sizeValue = this.RULE("sizeValue", () => {
    this.OR([
      { ALT: () => this.CONSUME(Small) },
      { ALT: () => this.CONSUME(Medium) },
      { ALT: () => this.CONSUME(Large) },
    ]);
  });

  // color_value ::= "morado" | "verde" | "rojo" | "naranja" | "azul" | "amarillo"
  private colorValue = this.RULE("colorValue", () => {
    this.OR([
      { ALT: () => this.CONSUME(Purple) },
      { ALT: () => this.CONSUME(Green) },
      { ALT: () => this.CONSUME(Red) },
      { ALT: () => this.CONSUME(Orange) },
      { ALT: () => this.CONSUME(Blue) },
      { ALT: () => this.CONSUME(Yellow) },
    ]);
  });

  // other_literal ::= category | type | shape_type | size_value | food_type | color_value
  private otherLiteral = this.RULE("otherLiteral", () => {
    this.OR([
      // Categories
      { ALT: () => this.CONSUME(Abstract) },
      { ALT: () => this.CONSUME(Pictorial) },
      { ALT: () => this.CONSUME(Concrete) },
      // Types
      { ALT: () => this.CONSUME(RationalType) },
      { ALT: () => this.CONSUME(ShapeType) },
      { ALT: () => this.CONSUME(FoodType) },
      { ALT: () => this.CONSUME(MontessoriType) },
      // Shape subtypes
      { ALT: () => this.CONSUME(Circle) },
      { ALT: () => this.CONSUME(Square) },
      { ALT: () => this.CONSUME(Triangle) },
      { ALT: () => this.CONSUME(Rectangle) },
      { ALT: () => this.CONSUME(Diamond) },
      { ALT: () => this.CONSUME(Star) },
      { ALT: () => this.CONSUME(Trapezoid) },
      // Size values
      { ALT: () => this.CONSUME(Small) },
      { ALT: () => this.CONSUME(Medium) },
      { ALT: () => this.CONSUME(Large) },
      // Food subtypes
      { ALT: () => this.CONSUME(Grape) },
      { ALT: () => this.CONSUME(Pear) },
      { ALT: () => this.CONSUME(Apple) },
      { ALT: () => this.CONSUME(Burger) },
      { ALT: () => this.CONSUME(Pasta) },
      // Color values
      { ALT: () => this.CONSUME(Purple) },
      { ALT: () => this.CONSUME(Green) },
      { ALT: () => this.CONSUME(Red) },
      { ALT: () => this.CONSUME(Orange) },
      { ALT: () => this.CONSUME(Blue) },
      { ALT: () => this.CONSUME(Yellow) },
    ]);
  });
}

// Singleton parser instance
export const parserInstance = new DataflowParser();
