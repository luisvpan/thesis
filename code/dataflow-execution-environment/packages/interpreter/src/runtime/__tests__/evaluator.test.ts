// §2.3 Evaluación dirigida por demanda y §2.4 determinismo

import { describe, expect, test } from "bun:test";
import { Interpreter } from "../../index";
import { numberLiteral, only } from "../../__tests__/helpers";

const num = numberLiteral;

describe("Evaluación por demanda", () => {
  test("memoiza: cada nodo se evalúa una sola vez", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source x = ${num(5)};
      transform a = sum(x, x);
      transform b = sum(a, a);
      transform c = sum(b, b);
      sink result = c;
    `);

    expect(result.errors).toHaveLength(0);
    // 5 + 5 = 10, 10 + 10 = 20, 20 + 20 = 40
    expect(only(result.results.get("result")!)).toBe("40");
  });

  test("expone los transforms ya calculados", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source x = ${num(2)};
      source y = ${num(3)};
      transform mid = sum(x, y);
      sink result = mid;
    `);

    expect(result.errors).toHaveLength(0);
    expect(only(result.results.get("mid")!)).toBe("5");
  });

  test("un error en una salida no impide el valor de las demás", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = ${num(10)};
      source cero = ${num(0)};
      transform malo = divide(a, cero);
      transform bueno = sum(a, a);
      sink roto = malo;
      sink sano = bueno;
    `);

    expect(result.errors).toHaveLength(1);
    expect(result.results.has("roto")).toBe(false);
    expect(only(result.results.get("sano")!)).toBe("20");
  });

  test("un nodo incompleto evalúa a nulo y no invalida el resto", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = ${num(4)};
      source incompleto = ;
      transform total = sum(a, incompleto);
      sink result = total;
      sink vacio = incompleto;
    `);

    expect(result.errors).toHaveLength(0);
    expect(only(result.results.get("result")!)).toBe("4");
    expect(result.results.get("vacio")).toEqual({ kind: "bolsa", entries: [] });
  });
});

describe("Reevaluación incremental", () => {
  test("la primera ejecución evalúa todos los nodos", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = ${num(5)};
      source b = ${num(3)};
      transform sum_ab = sum(a, b);
      sink result = sum_ab;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(4);
    expect(stats.evaluated).toBe(4);
    expect(stats.cached).toBe(0);
  });

  test("reutiliza la caché de los nodos que no cambiaron", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = ${num(5)};
      source b = ${num(3)};
      transform sum_ab = sum(a, b);
      sink result = sum_ab;
    `);

    const result = await interpreter.execute(`
      source a = ${num(5)};
      source b = ${num(10)};
      transform sum_ab = sum(a, b);
      sink result = sum_ab;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.cached).toBe(1); // 'a' se reutiliza
    expect(stats.evaluated).toBe(3);
    expect(only(result.results.get("result")!)).toBe("15");
  });

  test("invalida en cascada a los dependientes", async () => {
    const interpreter = new Interpreter();
    const program = (x: number) => `
      source x = ${num(x)};
      source dos = ${num(2)};
      transform doubled = multiply(x, dos);
      transform quadrupled = multiply(doubled, dos);
      sink result = quadrupled;
    `;

    await interpreter.execute(program(2));
    const result = await interpreter.execute(program(5));

    const stats = interpreter.getEvaluationStats();
    expect(stats.cached).toBe(1); // solo 'dos' sobrevive
    expect(stats.evaluated).toBe(4);
    expect(only(result.results.get("result")!)).toBe("20");
  });

  test("admite nodos añadidos", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = ${num(5)};
      sink result = a;
    `);

    const result = await interpreter.execute(`
      source a = ${num(5)};
      source b = ${num(3)};
      transform total = sum(a, b);
      sink result = total;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(4);
    expect(stats.cached).toBe(1);
    expect(stats.evaluated).toBe(3);
    expect(only(result.results.get("result")!)).toBe("8");
  });

  test("admite nodos eliminados", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`
      source a = ${num(5)};
      source b = ${num(3)};
      transform total = sum(a, b);
      sink result = total;
    `);

    const result = await interpreter.execute(`
      source a = ${num(5)};
      sink result = a;
    `);

    const stats = interpreter.getEvaluationStats();
    expect(stats.total).toBe(2);
    expect(stats.cached).toBe(1);
    expect(stats.evaluated).toBe(1);
    expect(only(result.results.get("result")!)).toBe("5");
  });

  test("reset() limpia el estado", async () => {
    const interpreter = new Interpreter();

    await interpreter.execute(`source x = ${num(100)}; sink result = x;`);
    expect(interpreter.getCacheSize()).toBe(2);

    interpreter.reset();
    expect(interpreter.getCacheSize()).toBe(0);
    expect(interpreter.getEvaluationStats()).toEqual({ evaluated: 0, cached: 0, total: 0 });

    const result = await interpreter.execute(`source y = ${num(1)}; sink result = y;`);
    expect(only(result.results.get("result")!)).toBe("1");
  });

  test("dos intérpretes son independientes", async () => {
    const one = new Interpreter();
    const other = new Interpreter();

    await one.execute(`source x = ${num(100)}; sink result = x;`);
    await other.execute(`source y = ${num(200)}; sink result = y;`);

    const result1 = await one.execute(`source x = ${num(101)}; sink result = x;`);
    const result2 = await other.execute(`source y = ${num(201)}; sink result = y;`);

    expect(one.getEvaluationStats().evaluated).toBe(2);
    expect(other.getEvaluationStats().evaluated).toBe(2);
    expect(only(result1.results.get("result")!)).toBe("101");
    expect(only(result2.results.get("result")!)).toBe("201");
  });
});
