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

// Las operaciones no son palabras clave: `operation ::= identifier` (§5.1), y el
// conjunto reconocido lo valida la pasada estática (§4.2.4).

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

// rational_literal ::= "-"? digit+ ( "/" digit+ | "." digit+ )?   (§5.1)
// Un entero (3), una fracción (1/3) o un decimal (2.5): las dos formas son
// alternativas excluyentes, y todo se interpreta como un racional exacto.
export const NumberLiteral = createToken({
  name: "NumberLiteral",
  pattern: /-?[0-9]+(\/[0-9]+|\.[0-9]+)?/,
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
