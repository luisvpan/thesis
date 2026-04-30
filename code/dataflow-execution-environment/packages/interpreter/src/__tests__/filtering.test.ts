import { describe, test, expect } from "bun:test";
import { Interpreter } from "../index";
import type { ArrayValue } from "../runtime/types";

describe("Filter operation (integration)", () => {
  test("filters shapes by size", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source shapes = [
        {type: circulo, size: grande, amount: 1},
        {type: cuadrado, size: pequeño, amount: 1},
        {type: cuadrado, size: grande, amount: 1}
      ];
      transform large_shapes = filter(shapes, grande);
      sink result = large_shapes;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    expect(sinkResult.elements).toHaveLength(2);
  });
});
