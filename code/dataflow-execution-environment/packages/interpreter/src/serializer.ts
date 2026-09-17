// Conversión entre el texto de un programa y su forma estructurada.
//
// `serialize` lee texto y devuelve un `Program` (cantidades ya en `Fraction`);
// `deserialize` hace el camino inverso. La forma JSON-esca del texto es la
// gramática de §5.

import Fraction from "fraction.js";
import type {
  CriterionLiteral as ASTCriterionLiteral,
  DataLiteral as ASTDataLiteral,
  Literal as ASTLiteral,
  Program as ASTProgram,
  Statement as ASTStatement,
} from "./analyzer/ast";
import { DataflowLexer } from "./analyzer/lexer";
import { parserInstance } from "./analyzer/parser";
import { DataflowSyntaxError } from "./analyzer/syntax-error";
import { visitorInstance } from "./analyzer/visitor";
import type {
  BagLiteral,
  CriterionLiteral,
  Literal,
  Program,
  Statement,
} from "./program";
import { ImmutableBag } from "./bag-builder";
import type { CPACategory, Entry } from "./runtime/types";

export interface ParseError {
  message: string;
  line?: number;
  column?: number;
}

export interface SerializeResult {
  program: Program | null;
  errors: ParseError[];
}

/** Texto → AST. Los errores de sintaxis se devuelven con su posición (§4.1). */
export function parseToAst(input: string): { ast: ASTProgram | null; errors: ParseError[] } {
  const lexResult = DataflowLexer.tokenize(input);

  if (lexResult.errors.length > 0) {
    return {
      ast: null,
      errors: lexResult.errors.map((error) => ({
        message: error.message,
        line: error.line,
        column: error.column,
      })),
    };
  }

  parserInstance.input = lexResult.tokens;
  const cst = parserInstance.program();

  if (parserInstance.errors.length > 0) {
    return {
      ast: null,
      errors: parserInstance.errors.map((error) => ({
        message: error.message,
        line: error.token.startLine,
        column: error.token.startColumn,
      })),
    };
  }

  try {
    return { ast: visitorInstance.visit(cst) as ASTProgram, errors: [] };
  } catch (err) {
    if (err instanceof DataflowSyntaxError) {
      return { ast: null, errors: [{ message: err.message, line: err.line, column: err.column }] };
    }
    throw err;
  }
}

export function serialize(input: string): SerializeResult {
  const { ast, errors } = parseToAst(input);
  return ast ? { program: astToProgram(ast), errors: [] } : { program: null, errors };
}

export function deserialize(program: Program): string {
  return program.statements.map(deserializeStatement).join("\n");
}

// =============================================================================
// AST (texto) → Program (Fraction)
// =============================================================================

export function astToProgram(ast: ASTProgram): Program {
  return { type: "Program", statements: ast.statements.map(astStatementToStatement) };
}

function astStatementToStatement(stmt: ASTStatement): Statement {
  switch (stmt.type) {
    case "SourceStatement":
      return {
        type: "SourceStatement",
        identifier: stmt.identifier,
        // Un `source` sin valor es `nulo`: la bolsa vacía.
        value: stmt.value ? astLiteralToLiteral(stmt.value) : ImmutableBag.of([]),
      };
    case "TransformStatement":
      return {
        type: "TransformStatement",
        identifier: stmt.identifier,
        operation: stmt.operation ?? "",
        arguments: stmt.arguments.map((argument) => ({ type: "Identifier", name: argument.name })),
      };
    case "SinkStatement":
      return {
        type: "SinkStatement",
        identifier: stmt.identifier,
        sourceIdentifier: stmt.sourceIdentifier ?? "",
      };
  }
}

function astLiteralToLiteral(literal: ASTLiteral): Literal {
  switch (literal.type) {
    case "DataLiteral":
      return ImmutableBag.of([astDataLiteralToEntry(literal)]);
    case "GroupLiteral":
      return ImmutableBag.of(literal.elements.map(astDataLiteralToEntry));
    case "CriterionLiteral":
      return astCriterionLiteralToCriterionLiteral(literal);
  }
}

function astDataLiteralToEntry(literal: ASTDataLiteral): Entry {
  const attributes: Record<string, string> = {};
  for (const property of literal.attributes) {
    attributes[property.key] = Array.isArray(property.value) ? property.value[0] : property.value;
  }

  return {
    category: literal.category as CPACategory,
    type: literal.objType,
    subtype: literal.subtype,
    attributes,
    quantity: new Fraction(literal.quantity || "1"),
  };
}

function astCriterionLiteralToCriterionLiteral(literal: ASTCriterionLiteral): CriterionLiteral {
  const values: Record<string, string | string[]> = {};
  for (const property of literal.values) {
    values[property.key] = property.value;
  }

  return {
    type: "CriterionLiteral",
    sourceType: literal.sourceType,
    properties: [...literal.properties],
    values,
  };
}

// =============================================================================
// Program (Fraction) → texto
// =============================================================================

function deserializeStatement(stmt: Statement): string {
  switch (stmt.type) {
    case "SourceStatement":
      return `source ${stmt.identifier} = ${deserializeLiteral(stmt.value)};`;
    case "TransformStatement":
      return `transform ${stmt.identifier} = ${stmt.operation}(${stmt.arguments
        .map((argument) => argument.name)
        .join(", ")});`;
    case "SinkStatement":
      return `sink ${stmt.identifier} = ${stmt.sourceIdentifier};`;
  }
}

function deserializeLiteral(literal: Literal): string {
  return literal.type === "CriterionLiteral"
    ? deserializeCriterion(literal)
    : deserializeBag(literal);
}

/** Una entrada se escribe como objeto; 0 o varias, como grupo. */
function deserializeBag(literal: BagLiteral): string {
  if (literal.entries.length === 1) return deserializeEntry(literal.entries[0]);
  return `[${literal.entries.map(deserializeEntry).join(", ")}]`;
}

function deserializeEntry(entry: Entry): string {
  const properties = [
    `"sourceType": "data"`,
    `"category": "${entry.category}"`,
    `"type": "${entry.type}"`,
    `"subtype": "${entry.subtype}"`,
    `"quantity": ${fractionToString(entry.quantity)}`,
  ];

  for (const [key, value] of Object.entries(entry.attributes)) {
    properties.push(`"${key}": "${value}"`);
  }

  return `{${properties.join(", ")}}`;
}

function deserializeCriterion(criterion: CriterionLiteral): string {
  const properties = [
    `"sourceType": "${criterion.sourceType}"`,
    `"properties": [${criterion.properties.map((property) => `"${property}"`).join(", ")}]`,
  ];

  for (const [key, value] of Object.entries(criterion.values)) {
    properties.push(
      Array.isArray(value)
        ? `"${key}": [${value.map((item) => `"${item}"`).join(", ")}]`
        : `"${key}": "${String(value)}"`
    );
  }

  return `{${properties.join(", ")}}`;
}

/**
 * Un racional se escribe como entero o como fracción `n/d`: nunca como decimal,
 * que perdería exactitud (`1/3` no es `0.333…`).
 */
function fractionToString(value: Fraction): string {
  const sign = value.s < 0n ? "-" : "";
  return value.d === 1n ? `${sign}${value.n}` : `${sign}${value.n}/${value.d}`;
}
