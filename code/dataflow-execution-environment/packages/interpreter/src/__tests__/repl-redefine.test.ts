import { describe, test, expect } from "bun:test";
import { Interpreter } from "../index";
import type { CPAObject } from "../runtime/types";
import Fraction from "fraction.js";

describe("REPL redefinition behavior", () => {
  test("allows redefining source values using Map-based statements", async () => {
    const interpreter = new Interpreter();
    const statements = new Map<string, string>();
    const sinkNames = new Set<string>();

    function extractId(line: string): string | null {
      const match = line.match(/^(?:source|transform|sink)\s+(\w+)\s*=/);
      return match ? match[1] : null;
    }

    async function addStatement(line: string) {
      const id = extractId(line);
      if (!id) return null;

      const isSink = line.startsWith("sink ");
      statements.set(id, line); // Replaces if exists
      if (isSink) sinkNames.add(id);

      const program = Array.from(statements.values()).join("\n");
      return await interpreter.execute(program);
    }

    // Initial setup
    await addStatement('source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};');
    await addStatement('source y = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};');
    await addStatement("transform suma = sum(x, y);");
    const result1 = await addStatement("sink resultado = suma;");

    expect(result1?.errors).toHaveLength(0);
    expect((result1!.results.get("resultado") as CPAObject).quantity.equals(new Fraction(8))).toBe(true);

    // Redefine x
    const result2 = await addStatement('source x = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 10};');
    expect(result2?.errors).toHaveLength(0);
    expect((result2!.results.get("resultado") as CPAObject).quantity.equals(new Fraction(13))).toBe(true);

    // Redefine y
    const result3 = await addStatement('source y = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 20};');
    expect(result3?.errors).toHaveLength(0);
    expect((result3!.results.get("resultado") as CPAObject).quantity.equals(new Fraction(30))).toBe(true);
  });
});
