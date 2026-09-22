// El REPL redefine sentencias por identificador: reejecuta el programa entero
// y el valor de las salidas se actualiza en cascada.

import { describe, expect, test } from "bun:test";
import { Interpreter } from "../index";
import { numberLiteral, only } from "./helpers";

const num = numberLiteral;

describe("Redefinición en el REPL", () => {
  test("redefinir una fuente actualiza las salidas que dependen de ella", async () => {
    const interpreter = new Interpreter();
    const statements = new Map<string, string>();

    async function add(line: string) {
      const id = line.match(/^(?:source|transform|sink)\s+(\w+)\s*=/)?.[1];
      if (!id) throw new Error(`sentencia no reconocida: ${line}`);
      statements.set(id, line); // redefine si ya existía
      return interpreter.execute(Array.from(statements.values()).join("\n"));
    }

    await add(`source x = ${num(5)};`);
    await add(`source y = ${num(3)};`);
    await add("transform suma = sum(x, y);");
    const first = await add("sink resultado = suma;");

    expect(first.errors).toHaveLength(0);
    expect(only(first.results.get("resultado")!)).toBe("8");

    const second = await add(`source x = ${num(10)};`);
    expect(second.errors).toHaveLength(0);
    expect(only(second.results.get("resultado")!)).toBe("13");

    const third = await add(`source y = ${num(20)};`);
    expect(third.errors).toHaveLength(0);
    expect(only(third.results.get("resultado")!)).toBe("30");
  });
});
