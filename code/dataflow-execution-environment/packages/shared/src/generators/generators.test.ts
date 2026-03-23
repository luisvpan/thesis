import { describe, it, expect } from "bun:test";
import { getGenerator, hasGenerator, BUILTIN_GENERATORS } from "@dataflow/shared/generators";

describe("Generators Registry", () => {
  describe("Registry Functions", () => {
    it("hasGenerator should return true for all builtin generators", () => {
      expect(hasGenerator("counter")).toBe(true);
      expect(hasGenerator("range")).toBe(true);
      expect(hasGenerator("constant")).toBe(true);
      expect(hasGenerator("repeat")).toBe(true);
      expect(hasGenerator("cycle")).toBe(true);
    });

    it("hasGenerator should return false for unknown generators", () => {
      expect(hasGenerator("unknown")).toBe(false);
      expect(hasGenerator("")).toBe(false);
    });

    it("getGenerator should return factory for builtin generators", () => {
      expect(getGenerator("counter")).toBeDefined();
      expect(getGenerator("range")).toBeDefined();
      expect(getGenerator("constant")).toBeDefined();
      expect(getGenerator("repeat")).toBeDefined();
      expect(getGenerator("cycle")).toBeDefined();
    });

    it("getGenerator should return undefined for unknown generators", () => {
      expect(getGenerator("unknown")).toBeUndefined();
      expect(getGenerator("")).toBeUndefined();
    });
  });

  describe("counter generator", () => {
    it("should generate infinite sequence 0, 1, 2, 3, ...", () => {
      const factory = getGenerator("counter");
      expect(factory).toBeDefined();
      
      const gen = factory!();
      
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(2);
      expect(gen.next().value).toBe(3);
      expect(gen.next().value).toBe(4);
      expect(gen.next().value).toBe(5);
    });

    it("should create independent generators", () => {
      const factory = getGenerator("counter");
      
      const gen1 = factory!();
      const gen2 = factory!();
      
      expect(gen1.next().value).toBe(0);
      expect(gen2.next().value).toBe(0);
      
      expect(gen1.next().value).toBe(1);
      expect(gen2.next().value).toBe(1);
      
      expect(gen1.next().value).toBe(2);
      expect(gen2.next().value).toBe(2);
    });
  });

  describe("range generator", () => {
    it("should generate 0-9, then repeat 9", () => {
      const factory = getGenerator("range");
      expect(factory).toBeDefined();
      
      const gen = factory!();
      
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(2);
      expect(gen.next().value).toBe(3);
      expect(gen.next().value).toBe(4);
      expect(gen.next().value).toBe(5);
      expect(gen.next().value).toBe(6);
      expect(gen.next().value).toBe(7);
      expect(gen.next().value).toBe(8);
      expect(gen.next().value).toBe(9);
      
      expect(gen.next().value).toBe(9);
      expect(gen.next().value).toBe(9);
      expect(gen.next().value).toBe(9);
    });

    it("should create independent generators", () => {
      const factory = getGenerator("range");
      
      const gen1 = factory!();
      const gen2 = factory!();
      
      expect(gen1.next().value).toBe(0);
      expect(gen2.next().value).toBe(0);
      
      expect(gen1.next().value).toBe(1);
      expect(gen2.next().value).toBe(1);
      
      for (let i = 0; i < 15; i++) {
        gen1.next();
      }
      
      expect(gen1.next().value).toBe(9);
      expect(gen2.next().value).toBe(2);
    });
  });

  describe("constant generator", () => {
    it("should always return 0", () => {
      const factory = getGenerator("constant");
      expect(factory).toBeDefined();
      
      const gen = factory!();
      
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(0);
    });

    it("should create independent generators", () => {
      const factory = getGenerator("constant");
      
      const gen1 = factory!();
      const gen2 = factory!();
      
      expect(gen1.next().value).toBe(0);
      expect(gen2.next().value).toBe(0);
      
      gen1.next();
      gen1.next();
      
      expect(gen1.next().value).toBe(0);
      expect(gen2.next().value).toBe(0);
    });
  });

  describe("repeat generator", () => {
    it("should always return 1", () => {
      const factory = getGenerator("repeat");
      expect(factory).toBeDefined();
      
      const gen = factory!();
      
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(1);
    });

    it("should create independent generators", () => {
      const factory = getGenerator("repeat");
      
      const gen1 = factory!();
      const gen2 = factory!();
      
      expect(gen1.next().value).toBe(1);
      expect(gen2.next().value).toBe(1);
      
      gen1.next();
      gen1.next();
      
      expect(gen1.next().value).toBe(1);
      expect(gen2.next().value).toBe(1);
    });
  });

  describe("cycle generator", () => {
    it("should cycle through [0, 1] repeatedly", () => {
      const factory = getGenerator("cycle");
      expect(factory).toBeDefined();
      
      const gen = factory!();
      
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(1);
      expect(gen.next().value).toBe(0);
      expect(gen.next().value).toBe(1);
    });

    it("should create independent generators", () => {
      const factory = getGenerator("cycle");
      
      const gen1 = factory!();
      const gen2 = factory!();
      
      expect(gen1.next().value).toBe(0);
      expect(gen2.next().value).toBe(0);
      
      expect(gen1.next().value).toBe(1);
      expect(gen2.next().value).toBe(1);
      
      expect(gen1.next().value).toBe(0);
      expect(gen2.next().value).toBe(0);
    });
  });

  describe("Integration tests", () => {
    it("should support all 5 generators in registry", () => {
      expect(Object.keys(BUILTIN_GENERATORS)).toHaveLength(5);
      expect(Object.keys(BUILTIN_GENERATORS)).toContain("counter");
      expect(Object.keys(BUILTIN_GENERATORS)).toContain("range");
      expect(Object.keys(BUILTIN_GENERATORS)).toContain("constant");
      expect(Object.keys(BUILTIN_GENERATORS)).toContain("repeat");
      expect(Object.keys(BUILTIN_GENERATORS)).toContain("cycle");
    });
  });
});
