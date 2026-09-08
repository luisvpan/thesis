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

// Identifier (for variable names)
// Supports Spanish characters (á, é, í, ó, ú, ñ, ü)
export const Identifier = createToken({
  name: "Identifier",
  pattern: /[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ][a-zA-Z0-9_\-áéíóúñüÁÉÍÓÚÑÜ]*/,
});

// Statement keywords
export const Source = createToken({ name: "Source", pattern: /source/, longer_alt: Identifier });
export const Transform = createToken({ name: "Transform", pattern: /transform/, longer_alt: Identifier });
export const Sink = createToken({ name: "Sink", pattern: /sink/, longer_alt: Identifier });

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
export const First = createToken({ name: "First", pattern: /first/, longer_alt: Identifier });
export const Last = createToken({ name: "Last", pattern: /last/, longer_alt: Identifier });
export const Count = createToken({ name: "Count", pattern: /count/, longer_alt: Identifier });
export const Compare = createToken({ name: "Compare", pattern: /compare/, longer_alt: Identifier });

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

// String literal (JSON-like syntax with double quotes)
export const StringLiteral = createToken({
  name: "StringLiteral",
  pattern: /"[^"]*"/,
});

// Number literal (supports decimals, sign, and fractions like "1/3")
export const NumberLiteral = createToken({
  name: "NumberLiteral",
  pattern: /-?[0-9]+(\.[0-9]+)?(\/[0-9]+)?/,
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
  First,
  Last,
  Count,
  Compare,

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
  StringLiteral,
  NumberLiteral,

  // Identifier (must be last among pattern tokens)
  Identifier,
];

export const DataflowLexer = new Lexer(allTokens);
