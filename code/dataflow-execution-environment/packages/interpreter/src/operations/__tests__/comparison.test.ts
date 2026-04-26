import { describe, test, expect } from "bun:test";
import { execute } from "../../index";
import type { ArrayValue } from "../../runtime/types";

describe("Comparison operations", () => {
  test("less_than filters by threshold", async () => {
    const result = await execute(`
      source a = 1;
      source b = 5;
      source c = 3;
      source threshold = 4;
      transform small_nums = less_than(a, b, c, threshold);
      sink result = small_nums;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("array");
    expect(sinkResult.elements).toHaveLength(2); // 1 and 3 are < 4
  });

  test("greater_than filters by threshold", async () => {
    const result = await execute(`
      source a = 1;
      source b = 5;
      source c = 3;
      source threshold = 2;
      transform big_nums = greater_than(a, b, c, threshold);
      sink result = big_nums;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("array");
    expect(sinkResult.elements).toHaveLength(2); // 5 and 3 are > 2
  });
});
