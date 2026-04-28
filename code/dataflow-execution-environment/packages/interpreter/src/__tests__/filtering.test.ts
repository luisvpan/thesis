import { describe, test, expect } from "bun:test";
import { Interpreter } from "../index";
import type { ArrayValue } from "../runtime/types";

describe("Filter operation (integration)", () => {
  test("filters shapes by size", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source shapes = [
        {type: circle, size: large, amount: 1},
        {type: square, size: small, amount: 1},
        {type: square, size: large, amount: 1}
      ];
      transform large_shapes = filter(shapes, large);
      sink result = large_shapes;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("array");
    expect(sinkResult.elements).toHaveLength(2);
  });
});
