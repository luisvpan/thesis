import { describe, test, expect } from "bun:test";
import { Interpreter } from "../index";
import type { ArrayValue } from "../runtime/types";

describe("Comparison operations (integration)", () => {
  test("less_than filters by threshold", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      source c = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      source threshold = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 4};
      transform small_nums = less_than(a, b, c, threshold);
      sink result = small_nums;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    expect(sinkResult.elements).toHaveLength(2); // 1 and 3 are < 4
  });

  test("greater_than filters by threshold", async () => {
    const interpreter = new Interpreter();
    const result = await interpreter.execute(`
      source a = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 1};
      source b = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 5};
      source c = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 3};
      source threshold = {"category": "abstracto", "type": "numero", "subtype": "racional", "quantity": 2};
      transform big_nums = greater_than(a, b, c, threshold);
      sink result = big_nums;
    `);

    expect(result.errors).toHaveLength(0);
    const sinkResult = result.results.get("result") as ArrayValue;
    expect(sinkResult.kind).toBe("arreglo");
    expect(sinkResult.elements).toHaveLength(2); // 5 and 3 are > 2
  });
});
