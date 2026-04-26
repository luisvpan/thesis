import { createToken, Lexer } from "chevrotain";

// Whitespace and comments (skipped)
export const WhiteSpace = createToken({
  name: "WhiteSpace",
  pattern: /\s+/,
  group: Lexer.SKIPPED,
});

export const Comment = createToken({
  name: "Comment",
  pattern: /\/\*[^*]*\*+([^/*][^*]*\*+)*\//,
  group: Lexer.SKIPPED,
});

// Identifier (defined first, referenced by keywords via longer_alt)
export const Identifier = createToken({
  name: "Identifier",
  pattern: /[a-zA-Z][a-zA-Z0-9_-]*/,
});

// Statement keywords
export const Source = createToken({ name: "Source", pattern: /source/, longer_alt: Identifier });
export const Transform = createToken({ name: "Transform", pattern: /transform/, longer_alt: Identifier });
export const Sink = createToken({ name: "Sink", pattern: /sink/, longer_alt: Identifier });

// Category keywords
export const Abstract = createToken({ name: "Abstract", pattern: /abstract/, longer_alt: Identifier });
export const Pictorial = createToken({ name: "Pictorial", pattern: /pictorial/, longer_alt: Identifier });
export const Concrete = createToken({ name: "Concrete", pattern: /concrete/, longer_alt: Identifier });

// Type keywords (v2.1.0: rational instead of integer)
export const RationalType = createToken({ name: "RationalType", pattern: /rational/, longer_alt: Identifier });
export const ShapeType = createToken({ name: "ShapeType", pattern: /shape/, longer_alt: Identifier });
export const FoodType = createToken({ name: "FoodType", pattern: /food/, longer_alt: Identifier });

// Shape subtypes
export const Circle = createToken({ name: "Circle", pattern: /circle/, longer_alt: Identifier });
export const Square = createToken({ name: "Square", pattern: /square/, longer_alt: Identifier });

// Food subtypes
export const Grape = createToken({ name: "Grape", pattern: /grape/, longer_alt: Identifier });
export const Pear = createToken({ name: "Pear", pattern: /pear/, longer_alt: Identifier });
export const Apple = createToken({ name: "Apple", pattern: /apple/, longer_alt: Identifier });
export const Burger = createToken({ name: "Burger", pattern: /burger/, longer_alt: Identifier });

// Size values
export const Small = createToken({ name: "Small", pattern: /small/, longer_alt: Identifier });
export const Medium = createToken({ name: "Medium", pattern: /medium/, longer_alt: Identifier });
export const Large = createToken({ name: "Large", pattern: /large/, longer_alt: Identifier });

// Color values
export const Purple = createToken({ name: "Purple", pattern: /purple/, longer_alt: Identifier });
export const Green = createToken({ name: "Green", pattern: /green/, longer_alt: Identifier });
export const Red = createToken({ name: "Red", pattern: /red/, longer_alt: Identifier });
export const Orange = createToken({ name: "Orange", pattern: /orange/, longer_alt: Identifier });

// Operation keywords
export const Sum = createToken({ name: "Sum", pattern: /sum/, longer_alt: Identifier });
export const Substract = createToken({ name: "Substract", pattern: /substract/, longer_alt: Identifier });
export const Multiply = createToken({ name: "Multiply", pattern: /multiply/, longer_alt: Identifier });
export const Divide = createToken({ name: "Divide", pattern: /divide/, longer_alt: Identifier });
export const LessThan = createToken({ name: "LessThan", pattern: /less_than/, longer_alt: Identifier });
export const GreaterThan = createToken({ name: "GreaterThan", pattern: /greater_than/, longer_alt: Identifier });
export const OrderAsc = createToken({ name: "OrderAsc", pattern: /order_asc/, longer_alt: Identifier });
export const OrderDesc = createToken({ name: "OrderDesc", pattern: /order_desc/, longer_alt: Identifier });
export const Filter = createToken({ name: "Filter", pattern: /filter/, longer_alt: Identifier });

// Object property keywords
export const Category = createToken({ name: "Category", pattern: /category/, longer_alt: Identifier });
export const Type = createToken({ name: "Type", pattern: /type/, longer_alt: Identifier });
export const Value = createToken({ name: "Value", pattern: /value/, longer_alt: Identifier });
export const Subtype = createToken({ name: "Subtype", pattern: /subtype/, longer_alt: Identifier });
export const Size = createToken({ name: "Size", pattern: /size/, longer_alt: Identifier });
export const Amount = createToken({ name: "Amount", pattern: /amount/, longer_alt: Identifier });
export const Color = createToken({ name: "Color", pattern: /color/, longer_alt: Identifier });

// Symbols
export const Equals = createToken({ name: "Equals", pattern: /=/ });
export const Semicolon = createToken({ name: "Semicolon", pattern: /;/ });
export const Comma = createToken({ name: "Comma", pattern: /,/ });
export const Colon = createToken({ name: "Colon", pattern: /:/ });
export const LParen = createToken({ name: "LParen", pattern: /\(/ });
export const RParen = createToken({ name: "RParen", pattern: /\)/ });
export const LBrace = createToken({ name: "LBrace", pattern: /\{/ });
export const RBrace = createToken({ name: "RBrace", pattern: /\}/ });
export const LBracket = createToken({ name: "LBracket", pattern: /\[/ });
export const RBracket = createToken({ name: "RBracket", pattern: /\]/ });

// Number literal (v2.1.0: supports decimals and sign)
export const NumberLiteral = createToken({
  name: "NumberLiteral",
  pattern: /-?[0-9]+(\.[0-9]+)?/,
});

// Token order matters - keywords must come before Identifier
export const allTokens = [
  // Skipped tokens
  WhiteSpace,
  Comment,

  // Statement keywords
  Source,
  Transform,
  Sink,

  // Category keywords
  Abstract,
  Pictorial,
  Concrete,

  // Type keywords
  RationalType,
  ShapeType,
  FoodType,

  // Shape subtypes
  Circle,
  Square,

  // Food subtypes
  Grape,
  Pear,
  Apple,
  Burger,

  // Size values
  Small,
  Medium,
  Large,

  // Color values
  Purple,
  Green,
  Red,
  Orange,

  // Operations
  Sum,
  Substract,
  Multiply,
  Divide,
  LessThan,
  GreaterThan,
  OrderAsc,
  OrderDesc,
  Filter,

  // Property keywords
  Category,
  Type,
  Value,
  Subtype,
  Size,
  Amount,
  Color,

  // Symbols
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

  // Literals
  NumberLiteral,

  // Identifier (must be last among pattern tokens)
  Identifier,
];

export const DataflowLexer = new Lexer(allTokens);
