import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";

describe("Parser Set/Object Literal Disambiguation", () => {
  const compiler = new Compiler();

  describe("Set Literal Parsing", () => {
    it("should parse set literal with numbers", () => {
      const result = compiler.compile(`
        source nums: set<natural> = {1, 2, 3};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse set literal with strings", () => {
      const result = compiler.compile(`
        source fruits: set<text> = {"apple", "banana", "cherry"};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse empty set literal", () => {
      const result = compiler.compile(`
        source empty: set<natural> = {};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse set literal with single element", () => {
      const result = compiler.compile(`
        source single: set<natural> = {42};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse set literal with boolean values", () => {
      const result = compiler.compile(`
        source flags: set<boolean> = {true, false, true};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });
  });

  describe("Object Literal Parsing", () => {
    it("should parse Shape object literal", () => {
      const result = compiler.compile(`
        source myShape: shape = {type: "circle", size: "large", color: "red"};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse Car object literal", () => {
      const result = compiler.compile(`
        source myCar: car = {color: "red"};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse Food object literal", () => {
      const result = compiler.compile(`
        source myFood: food = {taste: "sweet", color: "red"};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse Animal object literal", () => {
      const result = compiler.compile(`
        source myAnimal: animal = {type: "dog", color: "brown"};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse Person object literal", () => {
      const result = compiler.compile(`
        source myPerson: person = {ageGroup: "child", gender: "female"};
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });
  });

  describe("Set of Objects", () => {
    it("should parse set of Shape objects", () => {
      const result = compiler.compile(`
        source shapes: set<shape> = {
          {type: "circle", size: "large", color: "red"},
          {type: "triangle", size: "small", color: "blue"}
        };
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should parse set of Car objects", () => {
      const result = compiler.compile(`
        source cars: set<car> = {
          {color: "red"},
          {color: "blue"},
          {color: "red"}
        };
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });
  });

  describe("Disambiguation Edge Cases", () => {
    it("should distinguish object {color: 'red'} from set {1, 2, 3}", () => {
      const objectResult = compiler.compile(`
        source myCar: car = {color: "red"};
      `);
      expect(objectResult.success).toBe(true);

      const setResult = compiler.compile(`
        source nums: set<natural> = {1, 2, 3};
      `);
      expect(setResult.success).toBe(true);
    });
  });

  describe("Integration with Operations", () => {
    it("should use object literals in FILTER operations", () => {
      const result = compiler.compile(`
        source shapes: set<shape> = {
          {type: "circle", size: "large", color: "red"},
          {type: "triangle", size: "small", color: "blue"}
        };
        transform redShapes: set<shape> = FILTER_BY_COLOR(shapes, "red");
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });

    it("should use set literals in UNION operations", () => {
      const result = compiler.compile(`
        source set1: set<natural> = {1, 2, 3};
        source set2: set<natural> = {4, 5, 6};
        transform union: set<natural> = UNION(set1, set2);
      `);
      expect(result.success).toBe(true);
      expect(result.program).toBeDefined();
    });
  });
});
