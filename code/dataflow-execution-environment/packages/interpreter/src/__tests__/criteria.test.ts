/**
 * Tests for v4.0.0 Criteria features:
 * - CriteriaLiteral parsing and evaluation
 * - Filter with OR of ANDs logic
 * - Order with RTL criteria compilation
 * - Arithmetic operations with criteria pass-through
 */
import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { serialize } from "../serializer";
import { filter } from "../operations/filtering";
import { orderAsc, orderDesc } from "../operations/ordering";
import { sum, multiply, substract, divide } from "../operations/arithmetic";
import type { CPAObject, CriteriaObject, ArrayValue } from "../runtime/types";

// Helper to create CPA objects
function createCPA(
  category: "concreto" | "pictorico" | "abstracto",
  type: string,
  subtype: string,
  quantity: number,
  attributes: Record<string, string> = {}
): CPAObject {
  return {
    kind: "cpa",
    category,
    type,
    subtype,
    quantity: new Fraction(quantity),
    attributes,
  };
}

// Helper to create criteria objects
function createCriteria(
  properties: string[],
  values: Record<string, string | string[]>
): CriteriaObject {
  return {
    kind: "criteria",
    properties,
    values,
  };
}

describe("v4.0.0 Criteria - Parser/AST", () => {
  test("parses DataLiteral with sourceType: data", () => {
    const code = `source x = { "sourceType": "data", "category": "concreto", "type": "comida", "subtype": "manzana" };`;
    const { program, errors } = serialize(code);
    expect(errors).toHaveLength(0);
    expect(program).not.toBeNull();

    const stmt = program!.statements[0];
    expect(stmt.type).toBe("SourceStatement");
    if (stmt.type === "SourceStatement") {
      expect(stmt.value.type).toBe("DataLiteral");
    }
  });

  test("parses CriteriaLiteral with sourceType: criteria", () => {
    const code = `source x = { "sourceType": "criteria", "properties": ["size"], "size": "grande" };`;
    const { program, errors } = serialize(code);
    expect(errors).toHaveLength(0);
    expect(program).not.toBeNull();
  });

  test("parses array values in kv_pair", () => {
    const code = `source x = { "sourceType": "criteria", "properties": ["tier"], "tier": ["premium", "regular"] };`;
    const { program, errors } = serialize(code);
    expect(errors).toHaveLength(0);
  });

  test("parses empty array in kv_pair", () => {
    const code = `source x = { "sourceType": "criteria", "properties": [] };`;
    const { program, errors } = serialize(code);
    expect(errors).toHaveLength(0);
  });
});

describe("v4.0.0 Criteria - Filter OR of ANDs", () => {
  const items: CPAObject[] = [
    createCPA("pictorico", "forma", "circulo", 1, { size: "grande", color: "rojo" }),
    createCPA("pictorico", "forma", "cuadrado", 1, { size: "pequeño", color: "azul" }),
    createCPA("pictorico", "forma", "triangulo", 1, { size: "grande", color: "azul" }),
    createCPA("pictorico", "forma", "rectangulo", 1, { size: "pequeño", color: "rojo" }),
  ];

  test("single criteria filters by property value", () => {
    const criterion = createCriteria(["size"], { size: "grande" });
    const result = filter([...items, criterion]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2);
    expect((result.elements[0] as CPAObject).subtype).toBe("circulo");
    expect((result.elements[1] as CPAObject).subtype).toBe("triangulo");
  });

  test("multiple separate criteria use OR logic", () => {
    // size=grande OR color=azul → circulo, triangulo, cuadrado
    const c1 = createCriteria(["size"], { size: "grande" });
    const c2 = createCriteria(["color"], { color: "azul" });
    const result = filter([...items, c1, c2]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(3); // circulo, cuadrado, triangulo
  });

  test("grouped criteria use AND logic", () => {
    // (size=grande AND color=azul) → only triangulo
    const c1 = createCriteria(["size"], { size: "grande" });
    const c2 = createCriteria(["color"], { color: "azul" });
    const group: ArrayValue = { kind: "arreglo", elements: [c1, c2] };
    const result = filter([...items, group]) as CPAObject;

    expect(result.kind).toBe("cpa");
    expect(result.subtype).toBe("triangulo");
  });

  test("OR of ANDs: group AND separate", () => {
    // (size=grande AND color=rojo) OR size=pequeño → circulo, cuadrado, rectangulo
    const c1 = createCriteria(["size"], { size: "grande" });
    const c2 = createCriteria(["color"], { color: "rojo" });
    const c3 = createCriteria(["size"], { size: "pequeño" });
    const group: ArrayValue = { kind: "arreglo", elements: [c1, c2] };
    const result = filter([...items, group, c3]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(3); // circulo, cuadrado, rectangulo
  });

  test("incomplete criteria are ignored (return all items)", () => {
    // Criteria without values for properties = incomplete
    const incomplete = createCriteria(["size"], {}); // Has property but no value
    const result = filter([...items, incomplete]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(4); // All items
  });

  test("criteria with array values (OR within property)", () => {
    // tier in ["premium", "regular"] - but using size for this test
    const criterion = createCriteria(["size"], { size: ["grande", "pequeño"] });
    const result = filter([...items, criterion]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(4); // All match either grande or pequeño
  });
});

describe("v4.0.0 Criteria - Order RTL Compilation", () => {
  const items: CPAObject[] = [
    createCPA("concreto", "comida", "manzana", 3, { color: "rojo" }),
    createCPA("concreto", "comida", "pera", 1, { color: "verde" }),
    createCPA("concreto", "comida", "uva", 5, { color: "morado" }),
    createCPA("concreto", "comida", "manzana", 2, { color: "verde" }),
  ];

  test("order_asc by single criteria property", () => {
    const criterion = createCriteria(["subtype"], { subtype: ["manzana", "pera", "uva"] });
    const result = orderAsc([...items, criterion]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    // Ordered by subtype sequence: manzana(3), manzana(2), pera(1), uva(5)
    const subtypes = result.elements.map(e => (e as CPAObject).subtype);
    expect(subtypes[0]).toBe("manzana");
    expect(subtypes[1]).toBe("manzana");
    expect(subtypes[2]).toBe("pera");
    expect(subtypes[3]).toBe("uva");
  });

  test("order_desc reverses order", () => {
    const criterion = createCriteria(["subtype"], { subtype: ["manzana", "pera", "uva"] });
    const result = orderDesc([...items, criterion]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    // Reversed order: uva, pera, manzana, manzana
    const subtypes = result.elements.map(e => (e as CPAObject).subtype);
    expect(subtypes[0]).toBe("uva");
    expect(subtypes[1]).toBe("pera");
    expect(subtypes[2]).toBe("manzana");
    expect(subtypes[3]).toBe("manzana");
  });

  test("RTL compilation: later criteria have priority", () => {
    // c1 says order by color, c2 says order by subtype
    // RTL means c2 (subtype) has priority
    const c1 = createCriteria(["color"], { color: ["rojo", "verde", "morado"] });
    const c2 = createCriteria(["subtype"], { subtype: ["uva", "pera", "manzana"] });
    const result = orderAsc([...items, c1, c2]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    // Primary sort by subtype (c2), secondary by color (c1)
    const subtypes = result.elements.map(e => (e as CPAObject).subtype);
    expect(subtypes[0]).toBe("uva");
    expect(subtypes[1]).toBe("pera");
    expect(subtypes[2]).toBe("manzana");
    expect(subtypes[3]).toBe("manzana");
  });

  test("incomplete criteria are ignored in ordering (no-op)", () => {
    const incomplete = createCriteria(["color"], {}); // No values
    const result = orderAsc([...items, incomplete]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    // Incomplete criteria = no-op, maintains original order
    expect(result.elements).toHaveLength(4);
    const subtypes = result.elements.map(e => (e as CPAObject).subtype);
    expect(subtypes[0]).toBe("manzana"); // First in original order
    expect(subtypes[1]).toBe("pera");
    expect(subtypes[2]).toBe("uva");
    expect(subtypes[3]).toBe("manzana"); // Last in original order
  });

  test("no criteria returns original order (no-op)", () => {
    const result = orderAsc([...items]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    // No criteria = no-op, maintains original order
    const subtypes = result.elements.map(e => (e as CPAObject).subtype);
    expect(subtypes[0]).toBe("manzana"); // quantity 3
    expect(subtypes[1]).toBe("pera");
    expect(subtypes[2]).toBe("uva");
    expect(subtypes[3]).toBe("manzana"); // quantity 2
  });
});

describe("v4.0.0 Criteria - Arithmetic Pass-through", () => {
  const item1 = createCPA("concreto", "comida", "manzana", 3);
  const item2 = createCPA("concreto", "comida", "manzana", 2);
  const criterion = createCriteria(["color"], { color: "rojo" });

  test("sum passes criteria through to end of result", () => {
    const result = sum([item1, item2, criterion]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2); // summed item + criteria
    expect((result.elements[0] as CPAObject).kind).toBe("cpa");
    expect((result.elements[0] as CPAObject).quantity.valueOf()).toBe(5);
    expect((result.elements[1] as CriteriaObject).kind).toBe("criteria");
  });

  test("multiply passes criteria through to end of result", () => {
    const scalar = createCPA("abstracto", "numero", "racional", 2);
    const result = multiply([item1, scalar, criterion]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2); // scaled item + criteria
    expect((result.elements[0] as CPAObject).quantity.valueOf()).toBe(6);
    expect((result.elements[1] as CriteriaObject).kind).toBe("criteria");
  });

  test("substract passes criteria through to end of result", () => {
    const groupB: ArrayValue = { kind: "arreglo", elements: [item2, criterion] };
    const result = substract([item1, groupB]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2); // diff + criteria
    expect((result.elements[0] as CPAObject).quantity.valueOf()).toBe(1);
    expect((result.elements[1] as CriteriaObject).kind).toBe("criteria");
  });

  test("divide passes criteria through to end of result", () => {
    const divisor = createCPA("abstracto", "numero", "racional", 3);
    const groupB: ArrayValue = { kind: "arreglo", elements: [divisor, criterion] };
    const result = divide([item1, groupB]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements).toHaveLength(2); // quotient + criteria
    expect((result.elements[0] as CPAObject).quantity.valueOf()).toBe(1);
    expect((result.elements[1] as CriteriaObject).kind).toBe("criteria");
  });

  test("operations without criteria return normal result", () => {
    const result = sum([item1, item2]) as CPAObject;

    expect(result.kind).toBe("cpa");
    expect(result.quantity.valueOf()).toBe(5);
  });
});

describe("v4.0.0 Criteria - Comparison Pass-through", () => {
  test("less_than passes criteria through (from comparison module)", async () => {
    const items: CPAObject[] = [
      createCPA("concreto", "comida", "manzana", 5),
      createCPA("concreto", "comida", "pera", 2),
      createCPA("concreto", "comida", "uva", 8),
    ];
    const threshold = createCPA("abstracto", "numero", "racional", 4);
    const criterion = createCriteria(["color"], { color: "rojo" });
    const group: ArrayValue = { kind: "arreglo", elements: [...items, criterion] };

    const { lessThan } = await import("../operations/comparison");
    // Items with quantity < 4: pera(2)
    const result = lessThan([group, threshold]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    // pera + criteria
    expect(result.elements.length).toBeGreaterThanOrEqual(1);
    const cpa = result.elements.find(e => (e as CPAObject).kind === "cpa") as CPAObject;
    expect(cpa.subtype).toBe("pera");
  });

  test("greater_than passes criteria through (from comparison module)", async () => {
    const items: CPAObject[] = [
      createCPA("concreto", "comida", "manzana", 5),
      createCPA("concreto", "comida", "pera", 2),
      createCPA("concreto", "comida", "uva", 8),
    ];
    const threshold = createCPA("abstracto", "numero", "racional", 4);
    const criterion = createCriteria(["color"], { color: "rojo" });
    const group: ArrayValue = { kind: "arreglo", elements: [...items, criterion] };

    const { greaterThan } = await import("../operations/comparison");
    // Items with quantity > 4: manzana(5), uva(8)
    const result = greaterThan([group, threshold]) as ArrayValue;

    expect(result.kind).toBe("arreglo");
    expect(result.elements.length).toBeGreaterThanOrEqual(2);
  });
});
