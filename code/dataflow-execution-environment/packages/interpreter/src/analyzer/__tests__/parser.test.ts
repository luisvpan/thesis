import { describe, test, expect } from "bun:test";
import { DataflowLexer } from "../lexer";
import { parserInstance } from "../parser";
import { visitorInstance } from "../visitor";
import type { Program, DataLiteral } from "../ast";

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

describe("Parser v4.0.0", () => {
  test("parses CPA abstracto (numbers as objects)", () => {
    const result = parse('source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};');
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value?.type).toBe("DataLiteral");
      const obj = stmt.value as DataLiteral;
      expect(obj.category).toBe("abstracto");
    }
  });

  test("parses CPA pictorico", () => {
    const result = parse('source shapes = {"category": "pictorico", "type": "forma", "subtype": "circulo", "quantity": 3};');
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value?.type).toBe("DataLiteral");
      const obj = stmt.value as DataLiteral;
      expect(obj.category).toBe("pictorico");
    }
  });

  test("parses CPA concreto", () => {
    const result = parse('source caps = {"category": "concreto", "type": "tapa", "subtype": "plastico", "quantity": 10};');
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value?.type).toBe("DataLiteral");
      const obj = stmt.value as DataLiteral;
      expect(obj.category).toBe("concreto");
    }
  });

  test("parses object with decimal quantity", () => {
    const result = parse('source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3.14};');
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value?.type).toBe("DataLiteral");
      const obj = stmt.value as DataLiteral;
      expect(obj.quantity).toBe("3.14");
    }
  });

  test("rejects bare number literals", () => {
    const result = parse("source x = 5;");
    expect(result.errors.length).toBeGreaterThan(0);
  });

  test("rejects negative number literals", () => {
    const result = parse("source x = -2.5;");
    expect(result.errors.length).toBeGreaterThan(0);
  });
});
