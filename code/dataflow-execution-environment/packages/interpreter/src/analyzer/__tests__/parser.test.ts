import { describe, test, expect } from "bun:test";
import { parse } from "../../index";

describe("Parser v2.1.0", () => {
  test("parses rational numbers (decimals)", () => {
    const result = parse("source x = 3.14;");
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value).toEqual({ type: "NumberLiteral", value: 3.14 });
    }
  });

  test("parses negative decimals", () => {
    const result = parse("source x = -2.5;");
    expect(result.errors).toHaveLength(0);
    const stmt = result.ast!.statements[0];
    if (stmt.type === "SourceStatement") {
      expect(stmt.value).toEqual({ type: "NumberLiteral", value: -2.5 });
    }
  });

  test("parses rational type in objects", () => {
    const result = parse("source x = {category: abstract, type: rational, value: 42};");
    expect(result.errors).toHaveLength(0);
  });
});
