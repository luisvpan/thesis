import { describe, test, expect } from "bun:test";
import Fraction from "fraction.js";
import { orderAsc, orderDesc } from "../operations/ordering";
import type { ArrayValue, CPAObject, CriteriaObject } from "../runtime/types";

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

describe("Ordering operations (unit)", () => {
  describe("No criteria = no-op (preserves original order)", () => {
    test("order_asc without criteria returns items in original order", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 3, { size: "grande" }),
        createCPA("concreto", "comida", "uva", 5, { color: "morado" }),
        createCPA("concreto", "comida", "manzana", 1, { color: "rojo" }),
      ];
      const result = orderAsc(items) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // Should maintain original order (no sorting without criteria)
      expect((result.elements[0] as CPAObject).subtype).toBe("circulo");
      expect((result.elements[1] as CPAObject).subtype).toBe("uva");
      expect((result.elements[2] as CPAObject).subtype).toBe("manzana");
    });

    test("order_desc without criteria returns items in original order", () => {
      const items = [
        createCPA("concreto", "comida", "manzana", 1, { color: "rojo" }),
        createCPA("pictorico", "forma", "circulo", 3, { size: "grande" }),
        createCPA("concreto", "comida", "uva", 5, { color: "morado" }),
      ];
      const result = orderDesc(items) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // Should maintain original order (no sorting without criteria)
      expect((result.elements[0] as CPAObject).subtype).toBe("manzana");
      expect((result.elements[1] as CPAObject).subtype).toBe("circulo");
      expect((result.elements[2] as CPAObject).subtype).toBe("uva");
    });

    test("single item without criteria returns that item", () => {
      const item = createCPA("pictorico", "forma", "circulo", 1, { size: "grande" });
      const result = orderAsc([item]) as CPAObject;

      expect(result.kind).toBe("cpa");
      expect(result.category).toBe("pictorico");
      expect(result.subtype).toBe("circulo");
    });
  });

  describe("With criteria = sort by criteria property", () => {
    test("order_asc with size criteria sorts by size sequence", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 1, { size: "grande" }),
        createCPA("pictorico", "forma", "cuadrado", 1, { size: "pequeño" }),
        createCPA("pictorico", "forma", "triangulo", 1, { size: "mediano" }),
      ];
      const sizeCriteria = createCriteria(["size"], { size: ["pequeño", "mediano", "grande"] });
      const result = orderAsc([...items, sizeCriteria]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // Ascending by size: pequeño < mediano < grande
      expect((result.elements[0] as CPAObject).attributes.size).toBe("pequeño");
      expect((result.elements[1] as CPAObject).attributes.size).toBe("mediano");
      expect((result.elements[2] as CPAObject).attributes.size).toBe("grande");
    });

    test("order_desc with size criteria sorts by size sequence in reverse", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 1, { size: "grande" }),
        createCPA("pictorico", "forma", "cuadrado", 1, { size: "pequeño" }),
        createCPA("pictorico", "forma", "triangulo", 1, { size: "mediano" }),
      ];
      const sizeCriteria = createCriteria(["size"], { size: ["pequeño", "mediano", "grande"] });
      const result = orderDesc([...items, sizeCriteria]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // Descending by size: grande > mediano > pequeño
      expect((result.elements[0] as CPAObject).attributes.size).toBe("grande");
      expect((result.elements[1] as CPAObject).attributes.size).toBe("mediano");
      expect((result.elements[2] as CPAObject).attributes.size).toBe("pequeño");
    });

    test("order_asc with color criteria sorts by color sequence", () => {
      const items = [
        createCPA("concreto", "comida", "uva", 1, { color: "morado" }),
        createCPA("concreto", "comida", "manzana", 1, { color: "rojo" }),
        createCPA("concreto", "comida", "pera", 1, { color: "verde" }),
      ];
      const colorCriteria = createCriteria(["color"], { color: ["verde", "rojo", "morado"] });
      const result = orderAsc([...items, colorCriteria]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // Ascending by color: verde < rojo < morado
      expect((result.elements[0] as CPAObject).attributes.color).toBe("verde");
      expect((result.elements[1] as CPAObject).attributes.color).toBe("rojo");
      expect((result.elements[2] as CPAObject).attributes.color).toBe("morado");
    });
  });

  describe("Stable sort (ties maintain original order)", () => {
    test("order_asc with ties preserves original order among equal items", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 1, { size: "mediano" }),
        createCPA("pictorico", "forma", "cuadrado", 2, { size: "mediano" }),
        createCPA("pictorico", "forma", "triangulo", 3, { size: "pequeño" }),
        createCPA("pictorico", "forma", "rombo", 4, { size: "mediano" }),
      ];
      const sizeCriteria = createCriteria(["size"], { size: ["pequeño", "mediano", "grande"] });
      const result = orderAsc([...items, sizeCriteria]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // pequeño first, then the 3 medianos in original order
      expect((result.elements[0] as CPAObject).subtype).toBe("triangulo"); // pequeño
      expect((result.elements[1] as CPAObject).subtype).toBe("circulo");   // mediano (first in original)
      expect((result.elements[1] as CPAObject).quantity.valueOf()).toBe(1);
      expect((result.elements[2] as CPAObject).subtype).toBe("cuadrado");  // mediano (second in original)
      expect((result.elements[3] as CPAObject).subtype).toBe("rombo");     // mediano (third in original)
      expect((result.elements[3] as CPAObject).quantity.valueOf()).toBe(4);
    });

    test("order_desc with ties preserves original order among equal items", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 1, { size: "grande" }),
        createCPA("pictorico", "forma", "cuadrado", 2, { size: "grande" }),
        createCPA("pictorico", "forma", "triangulo", 3, { size: "pequeño" }),
      ];
      const sizeCriteria = createCriteria(["size"], { size: ["pequeño", "mediano", "grande"] });
      const result = orderDesc([...items, sizeCriteria]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // grande first (in original order), then pequeño
      expect((result.elements[0] as CPAObject).subtype).toBe("circulo");   // grande (first)
      expect((result.elements[1] as CPAObject).subtype).toBe("cuadrado");  // grande (second)
      expect((result.elements[2] as CPAObject).subtype).toBe("triangulo"); // pequeño
    });
  });

  describe("Incomplete criteria = no-op", () => {
    test("order_asc with criteria without values is no-op", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 3, { size: "grande" }),
        createCPA("pictorico", "forma", "cuadrado", 1, { size: "pequeño" }),
      ];
      const incompleteCriteria = createCriteria(["size"], {}); // Has property but no value
      const result = orderAsc([...items, incompleteCriteria]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // Should maintain original order (incomplete criteria = no-op)
      expect((result.elements[0] as CPAObject).subtype).toBe("circulo");
      expect((result.elements[1] as CPAObject).subtype).toBe("cuadrado");
    });
  });

  describe("Empty input handling", () => {
    test("order_asc with empty array returns empty array", () => {
      const result = orderAsc([]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      expect(result.elements).toHaveLength(0);
    });
  });

  describe("RTL criteria compilation", () => {
    test("later criteria take priority over earlier ones", () => {
      const items = [
        createCPA("pictorico", "forma", "circulo", 1, { size: "grande" }),
        createCPA("pictorico", "forma", "cuadrado", 1, { size: "pequeño" }),
        createCPA("pictorico", "forma", "triangulo", 1, { size: "mediano" }),
      ];
      const wrongOrder = createCriteria(["size"], { size: ["grande", "mediano", "pequeño"] });
      const correctOrder = createCriteria(["size"], { size: ["pequeño", "mediano", "grande"] });
      const result = orderAsc([...items, wrongOrder, correctOrder]) as ArrayValue;

      expect(result.kind).toBe("arreglo");
      // The second criteria (correctOrder) should override the first
      expect((result.elements[0] as CPAObject).attributes.size).toBe("pequeño");
      expect((result.elements[1] as CPAObject).attributes.size).toBe("mediano");
      expect((result.elements[2] as CPAObject).attributes.size).toBe("grande");
    });
  });
});
