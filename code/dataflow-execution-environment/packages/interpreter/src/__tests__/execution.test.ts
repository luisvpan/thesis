// §2.4 Determinismo: el resultado depende solo del programa, no del intérprete
// que lo ejecuta ni de lo que se ejecutó antes.

import { describe, expect, test } from "bun:test";
import { Interpreter } from "../index";
import { numberLiteral, only } from "./helpers";

const num = numberLiteral;

async function value(program: string, sink = "result"): Promise<string> {
  const result = await new Interpreter().execute(program);
  expect(result.errors).toHaveLength(0);
  return only(result.results.get(sink)!);
}

describe("Ejecuciones independientes", () => {
  test("con valores distintos", async () => {
    const program = (x: number) => `
      source x = ${num(x)};
      source dos = ${num(2)};
      transform doubled = multiply(x, dos);
      sink result = doubled;
    `;

    expect(await value(program(10))).toBe("20");
    expect(await value(program(50))).toBe("100");
  });

  test("con sentencias añadidas", async () => {
    expect(await value(`source a = ${num(5)}; sink result = a;`)).toBe("5");
    expect(
      await value(`
        source a = ${num(5)};
        source b = ${num(3)};
        transform total = sum(a, b);
        sink result = total;
      `)
    ).toBe("8");
  });

  test("con la operación cambiada", async () => {
    const program = (operation: string) => `
      source a = ${num(10)};
      source b = ${num(2)};
      transform calc = ${operation}(a, b);
      sink result = calc;
    `;

    expect(await value(program("sum"))).toBe("12");
    expect(await value(program("multiply"))).toBe("20");
    expect(await value(program("substract"))).toBe("8");
    expect(await value(program("divide"))).toBe("5");
  });

  test("dos intérpretes no comparten estado", async () => {
    const a = new Interpreter();
    const b = new Interpreter();

    const resultA = await a.execute(`source x = ${num(100)}; sink result = x;`);
    const resultB = await b.execute(`source y = ${num(1)}; sink result = y;`);

    expect(only(resultA.results.get("result")!)).toBe("100");
    expect(only(resultB.results.get("result")!)).toBe("1");
  });

  test("el mismo programa da el mismo valor las veces que se ejecute", async () => {
    const interpreter = new Interpreter();
    const program = `
      source a = ${num("1/3")};
      source b = ${num("1/6")};
      transform total = sum(a, b);
      sink result = total;
    `;

    const first = await interpreter.execute(program);
    const second = await interpreter.execute(program);

    expect(only(first.results.get("result")!)).toBe("1/2");
    expect(only(second.results.get("result")!)).toBe("1/2");
  });
});
