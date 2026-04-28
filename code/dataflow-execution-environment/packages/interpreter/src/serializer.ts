// Serialization utilities for converting between dataflow strings and Program objects

import Fraction from "fraction.js";
import { DataflowLexer } from "./analyzer/lexer";
import { parserInstance } from "./analyzer/parser";
import { visitorInstance } from "./analyzer/visitor";
import type { Program as ASTProgram, Statement as ASTStatement, Literal as ASTLiteral, Expression as ASTExpression, ObjectLiteral as ASTObjectLiteral } from "./analyzer/ast";
import type {
  Program,
  Statement,
  Literal,
  Expression,
  ObjectLiteral,
  NumberLiteral,
  ArrayLiteral,
  OtherLiteral,
  SourceStatement,
  TransformStatement,
  SinkStatement,
} from "./program";

export interface ParseError {
  message: string;
  line?: number;
  column?: number;
}

export interface SerializeResult {
  program: Program | null;
  errors: ParseError[];
}

/**
 * Serializes a dataflow string into a Program object.
 * Converts string numeric values to Fraction for exact arithmetic.
 */
export function serialize(input: string): SerializeResult {
  const errors: ParseError[] = [];

  // Tokenize
  const lexResult = DataflowLexer.tokenize(input);

  if (lexResult.errors.length > 0) {
    for (const error of lexResult.errors) {
      errors.push({
        message: error.message,
        line: error.line,
        column: error.column,
      });
    }
    return { program: null, errors };
  }

  // Parse
  parserInstance.input = lexResult.tokens;
  const cst = parserInstance.program();

  if (parserInstance.errors.length > 0) {
    for (const error of parserInstance.errors) {
      errors.push({
        message: error.message,
        line: error.token.startLine,
        column: error.token.startColumn,
      });
    }
    return { program: null, errors };
  }

  // Build AST
  const ast = visitorInstance.visit(cst) as ASTProgram;

  // Convert AST to Program (string -> Fraction)
  const program = astToProgram(ast);

  return { program, errors: [] };
}

/**
 * Deserializes a Program object into a dataflow string.
 * Converts Fraction numeric values back to string representation.
 */
export function deserialize(program: Program): string {
  return program.statements.map(deserializeStatement).join("\n");
}

// Internal conversion: AST (string) -> Program (Fraction)

function astToProgram(ast: ASTProgram): Program {
  return {
    type: "Program",
    statements: ast.statements.map(astStatementToStatement),
  };
}

function astStatementToStatement(stmt: ASTStatement): Statement {
  switch (stmt.type) {
    case "SourceStatement":
      return {
        type: "SourceStatement",
        identifier: stmt.identifier,
        value: astLiteralToLiteral(stmt.value),
      };
    case "TransformStatement":
      return {
        type: "TransformStatement",
        identifier: stmt.identifier,
        operation: stmt.operation,
        arguments: stmt.arguments.map(astExpressionToExpression),
      };
    case "SinkStatement":
      return {
        type: "SinkStatement",
        identifier: stmt.identifier,
        sourceIdentifier: stmt.sourceIdentifier,
      };
  }
}

function astExpressionToExpression(expr: ASTExpression): Expression {
  if (expr.type === "Identifier") {
    return { type: "Identifier", name: expr.name };
  }
  return astLiteralToLiteral(expr);
}

function astLiteralToLiteral(lit: ASTLiteral): Literal {
  switch (lit.type) {
    case "NumberLiteral":
      return {
        type: "NumberLiteral",
        value: new Fraction(lit.value),
      };
    case "ArrayLiteral":
      return {
        type: "ArrayLiteral",
        elements: lit.elements.map(astExpressionToExpression),
      };
    case "OtherLiteral":
      return {
        type: "OtherLiteral",
        value: lit.value,
      };
    case "ObjectLiteral":
      return astObjectLiteralToObjectLiteral(lit);
  }
}

function astObjectLiteralToObjectLiteral(obj: ASTObjectLiteral): ObjectLiteral {
  const result: Record<string, unknown> = {
    type: "ObjectLiteral",
  };

  if (obj.category !== undefined) result.category = obj.category;
  if (obj.objectType !== undefined) result.objectType = obj.objectType;
  if ("subtype" in obj && obj.subtype !== undefined) result.subtype = obj.subtype;
  if ("size" in obj && obj.size !== undefined) result.size = obj.size;
  if ("color" in obj && obj.color !== undefined) result.color = obj.color;

  // Convert string numeric values to Fraction
  if ("amount" in obj && obj.amount !== undefined) result.amount = new Fraction(obj.amount);
  if ("value" in obj && obj.value !== undefined) result.value = new Fraction(obj.value);

  return result as ObjectLiteral;
}

// Internal conversion: Program (Fraction) -> dataflow string

function deserializeStatement(stmt: Statement): string {
  switch (stmt.type) {
    case "SourceStatement":
      return `source ${stmt.identifier} = ${deserializeLiteral(stmt.value)};`;
    case "TransformStatement":
      return `transform ${stmt.identifier} = ${stmt.operation}(${stmt.arguments.map(deserializeExpression).join(", ")});`;
    case "SinkStatement":
      return `sink ${stmt.identifier} = ${stmt.sourceIdentifier};`;
  }
}

function deserializeExpression(expr: Expression): string {
  if (expr.type === "Identifier") {
    return expr.name;
  }
  return deserializeLiteral(expr);
}

function deserializeLiteral(lit: Literal): string {
  switch (lit.type) {
    case "NumberLiteral":
      return fractionToString(lit.value);
    case "ArrayLiteral":
      return `[${lit.elements.map(deserializeExpression).join(", ")}]`;
    case "OtherLiteral":
      return lit.value;
    case "ObjectLiteral":
      return deserializeObjectLiteral(lit);
  }
}

function deserializeObjectLiteral(obj: ObjectLiteral): string {
  const props: string[] = [];

  if (obj.category !== undefined) props.push(`category: ${obj.category}`);
  if (obj.objectType !== undefined) props.push(`type: ${obj.objectType}`);
  if ("subtype" in obj && obj.subtype !== undefined) props.push(`subtype: ${obj.subtype}`);
  if ("size" in obj && obj.size !== undefined) props.push(`size: ${obj.size}`);
  if ("color" in obj && obj.color !== undefined) props.push(`color: ${obj.color}`);
  if ("amount" in obj && obj.amount !== undefined) props.push(`amount: ${fractionToString(obj.amount)}`);
  if ("value" in obj && obj.value !== undefined) props.push(`value: ${fractionToString(obj.value)}`);

  return `{${props.join(", ")}}`;
}

/**
 * Converts a Fraction to its string representation.
 * Uses decimal format for whole numbers and simple decimals,
 * or the original string representation for exact fractions.
 */
function fractionToString(f: Fraction): string {
  // If it's a whole number, return as integer
  if (f.d === 1) {
    return f.n.toString();
  }

  // Otherwise return as decimal
  return f.valueOf().toString();
}
