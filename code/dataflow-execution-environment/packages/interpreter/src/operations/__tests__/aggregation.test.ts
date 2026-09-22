// §3.6.1 — agregación

import { describe, expect, test } from "bun:test";
import { asNumber } from "../../runtime/bag";
import type { Bag } from "../../runtime/types";
import { bagOf, entry, only, pairs } from "../../__tests__/helpers";
import { count } from "../aggregation";
import { multiply } from "../arithmetic";

describe("count — §3.6.1", () => {
  test("suma las cantidades sin importar la identidad", () => {
    expect(only(count([bagOf(entry("manzana", 2), entry("pera", 3))]))).toBe("5");
  });

  test("no agrupa: totaliza también los repetidos", () => {
    expect(only(count([bagOf(entry("manzana", 2), entry("manzana", 4))]))).toBe("6");
  });

  test("suma racionales de forma exacta", () => {
    expect(only(count([bagOf(entry("manzana", "1/2"), entry("manzana", "1/2"))]))).toBe("1");
  });

  test("sobre nulo da el número 0", () => {
    const result = count([bagOf()]) as Bag;
    expect(only(result)).toBe("0");
    expect(result.entries[0].category).toBe("abstracto");
    expect(result.entries[0].type).toBe("numero");
  });

  test("el resultado es un número y puede alimentar a multiply", () => {
    const total = count([bagOf(entry("manzana", 2), entry("pera", 3))]) as Bag;
    expect(asNumber(total)?.valueOf()).toBe(5);
    expect(pairs(multiply([bagOf(entry("uva", 2)), total]))).toEqual(["uva:10"]);
  });
});
