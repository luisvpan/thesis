import { describe, test, expect } from "bun:test";
import { DataflowLexer } from "../lexer";
import { parserInstance } from "../parser";
import { visitorInstance } from "../visitor";
import type { Program } from "../ast";

interface ParseResult {
  ast: Program | null;
  errors: string[];
}

function parse(input: string): ParseResult {
  const lexResult = DataflowLexer.tokenize(input);
  if (lexResult.errors.length > 0) {
    return {
      ast: null,
      errors: lexResult.errors.map((e) => e.message),
    };
  }

  parserInstance.input = lexResult.tokens;
  const cst = parserInstance.program();

  if (parserInstance.errors.length > 0) {
    return {
      ast: null,
      errors: parserInstance.errors.map((e) => e.message),
    };
  }

  const ast = visitorInstance.visit(cst) as Program;
  return { ast, errors: [] };
}

describe("Parser v2.1.0", () => {
  test("parses rational numbers (decimals)", () => {
    const result = parse("source x = 3.14;");
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value).toEqual({ type: "NumberLiteral", value: "3.14" });
    }
  });

  test("parses negative decimals", () => {
    const result = parse("source x = -2.5;");
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value).toEqual({ type: "NumberLiteral", value: "-2.5" });
    }
  });

  test("parses rational type in objects", () => {
    const result = parse("source x = {category: abstracto, type: racional, value: 42};");
    expect(result.errors).toHaveLength(0);
  });
});
