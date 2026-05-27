// Serialization utilities for converting between dataflow strings and Program objects

import Fraction from "fraction.js";
import { DataflowLexer } from "./analyzer/lexer";
import { parserInstance } from "./analyzer/parser";
import { visitorInstance } from "./analyzer/visitor";
import type {
  Program as ASTProgram,
  Statement as ASTStatement,
  Literal as ASTLiteral,
  Expression as ASTExpression,
  ObjectLiteral as ASTObjectLiteral,
  DataLiteral as ASTDataLiteral,
  CriteriaLiteral as ASTCriteriaLiteral,
  ObjectProperty as ASTObjectProperty,
  GroupLiteral as ASTGroupLiteral,
} from "./analyzer/ast";
import type {
  Program,
  Statement,
  Literal,
  Expression,
  ObjectLiteral,
  ObjectProperty,
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
        value: stmt.value ? astLiteralToLiteral(stmt.value) : { type: "OtherLiteral", value: "" },
      };
    case "TransformStatement":
      return {
        type: "TransformStatement",
        identifier: stmt.identifier,
        operation: stmt.operation ?? "sum",
        arguments: stmt.arguments.map(astExpressionToExpression),
      };
    case "SinkStatement":
      return {
        type: "SinkStatement",
        identifier: stmt.identifier,
        sourceIdentifier: stmt.sourceIdentifier ?? "",
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
    case "ArrayLiteral":
      return {
        type: "ArrayLiteral",
        elements: lit.elements.map(astExpressionToExpression),
      };
    case "StringLiteral":
      return {
        type: "OtherLiteral",
        value: lit.value,
      };
    case "DataLiteral":
    case "CriteriaLiteral":
      return astObjectLiteralToObjectLiteral(lit);
    case "GroupLiteral":
      return {
        type: "ArrayLiteral",
        elements: lit.elements.map(astObjectLiteralToObjectLiteral),
      };
  }
}

function astObjectLiteralToObjectLiteral(obj: ASTObjectLiteral): ObjectLiteral {
  // Get raw properties based on object type
  const rawProps: ASTObjectProperty[] = obj.type === "DataLiteral"
    ? obj.attributes
    : obj.values;

  const properties: ObjectProperty[] = rawProps.map((prop: ASTObjectProperty) => {
    // Handle array values (e.g., "tier": ["premium", "regular"])
    if (Array.isArray(prop.value)) {
      return {
        key: prop.key,
        value: prop.value.join(","), // Serialize array as comma-separated string
      };
    }
    // Check if value looks like a number
    const numValue = parseFloat(prop.value);
    if (!isNaN(numValue) && prop.value.trim() === String(numValue)) {
      return {
        key: prop.key,
        value: new Fraction(prop.value),
      };
    }
    return {
      key: prop.key,
      value: prop.value,
    };
  });

  // For DataLiteral, add core properties
  if (obj.type === "DataLiteral") {
    // Add category, type, subtype as properties
    if (obj.category) properties.unshift({ key: "category", value: obj.category });
    if (obj.objType) properties.splice(1, 0, { key: "type", value: obj.objType });
    if (obj.subtype) properties.splice(2, 0, { key: "subtype", value: obj.subtype });
    if (obj.quantity && obj.quantity !== "1") {
      properties.push({ key: "quantity", value: new Fraction(obj.quantity) });
    }
  }

  // For CriteriaLiteral, add sourceType and properties array
  if (obj.type === "CriteriaLiteral") {
    properties.unshift({ key: "sourceType", value: "criteria" });
    properties.splice(1, 0, { key: "properties", value: obj.properties.join(",") });
  }

  // Add convenience quantity accessor
  const quantityProp = properties.find(p => p.key === "quantity");
  const quantity = quantityProp && quantityProp.value instanceof Fraction
    ? quantityProp.value
    : undefined;

  return {
    type: "ObjectLiteral",
    properties,
    quantity,
  };
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
    case "ArrayLiteral":
      return `[${lit.elements.map(deserializeExpression).join(", ")}]`;
    case "OtherLiteral":
      return `"${lit.value}"`;
    case "ObjectLiteral":
      return deserializeObjectLiteral(lit);
  }
}

function deserializeObjectLiteral(obj: ObjectLiteral): string {
  const props: string[] = obj.properties.map(prop => {
    if (prop.value instanceof Fraction) {
      return `"${prop.key}": ${fractionToString(prop.value)}`;
    }
    return `"${prop.key}": "${prop.value}"`;
  });

  return `{${props.join(", ")}}`;
}

/**
 * Converts a Fraction to its string representation.
 * Uses decimal format for whole numbers and simple decimals,
 * or the original string representation for exact fractions.
 */
function fractionToString(f: Fraction): string {
  // If it's a whole number, return as integer
  if (f.d === 1n) {
    return f.n.toString();
  }

  // Otherwise return as decimal
  return f.valueOf().toString();
}
