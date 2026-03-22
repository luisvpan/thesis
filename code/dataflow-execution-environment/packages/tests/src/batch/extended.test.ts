import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Curriculum Types Extended", () => {
  describe("Test 3.3: Filter Cars by Color", () => {
    it("should filter cars by color", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source cars: set<car> = {
          {color: "red"},
          {color: "blue"},
          {color: "red"},
          {color: "green"}
        };

        source target_color: text = "red";

        transform red_cars: set<car> = FILTER_BY_COLOR(cars, target_color);

        output result: set<car> = red_cars;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(2); // 2 red cars
    });

    it("should compare cars by color", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source car1: car = {color: "red"};
        source car2: car = {color: "blue"};

        transform same_color: boolean = COMPARE_BY_COLOR(car1, car2);

        output result: boolean = same_color;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: false }); // red ≠ blue
    });
  });

  describe("Test 3.4: Filter Foods by Color and Taste", () => {
    it("should filter foods by color", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source foods: set<food> = {
          {taste: "sweet", color: "red"},
          {taste: "salty", color: "blue"},
          {taste: "sweet", color: "red"},
          {taste: "bitter", color: "green"}
        };

        source target_color: text = "red";

        transform red_foods: set<food> = FILTER_BY_COLOR(foods, target_color);

        output result: set<food> = red_foods;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(2); // 2 red foods
    });

    it("should compare foods by color", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source food1: food = {taste: "sweet", color: "red"};
        source food2: food = {taste: "salty", color: "blue"};

        transform same_color: boolean = COMPARE_BY_COLOR(food1, food2);

        output result: boolean = same_color;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: false }); // red ≠ blue
    });
  });

  describe("Test 3.5: Filter Animals by Color and Type", () => {
    it("should filter animals by color", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source animals: set<animal> = {
          {type: "dog", color: "red"},
          {type: "cat", color: "blue"},
          {type: "bird", color: "red"},
          {type: "fish", color: "green"}
        };

        source target_color: text = "red";

        transform red_animals: set<animal> = FILTER_BY_COLOR(animals, target_color);

        output result: set<animal> = red_animals;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(2); // 2 red animals
    });

    it("should filter animals by type", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source animals: set<animal> = {
          {type: "dog", color: "red"},
          {type: "cat", color: "blue"},
          {type: "dog", color: "yellow"},
          {type: "bird", color: "green"}
        };

        source target_type: text = "dog";

        transform dogs: set<animal> = FILTER_BY_TYPE(animals, target_type);

        output result: set<animal> = dogs;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(2); // 2 dogs
    });

    it("should compare animals by color", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source animal1: animal = {type: "dog", color: "red"};
        source animal2: animal = {type: "cat", color: "blue"};

        transform same_color: boolean = COMPARE_BY_COLOR(animal1, animal2);

        output result: boolean = same_color;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: false }); // red ≠ blue
    });

    it("should compare animals by type", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source animal1: animal = {type: "dog", color: "red"};
        source animal2: animal = {type: "cat", color: "red"};

        transform same_type: boolean = COMPARE_BY_TYPE(animal1, animal2);

        output result: boolean = same_type;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "boolean", value: false }); // dog ≠ cat
    });
  });

  describe("Test 3.6: Set Operations on Curriculum Types", () => {
    it("should perform UNION on car sets", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source cars1: set<car> = {
          {color: "red"},
          {color: "blue"}
        };

        source cars2: set<car> = {
          {color: "red"},
          {color: "green"}
        };

        transform all_cars: set<car> = UNION(cars1, cars2);

        output result: set<car> = all_cars;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(3); // red, blue, green (red deduplicated)
    });

    it("should perform INTERSECTION on food sets", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source foods1: set<food> = {
          {taste: "sweet", color: "red"},
          {taste: "salty", color: "blue"}
        };

        source foods2: set<food> = {
          {taste: "sweet", color: "red"},
          {taste: "bitter", color: "green"}
        };

        transform common_foods: set<food> = INTERSECTION(foods1, foods2);

        output result: set<food> = common_foods;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(1); // only the common food
    });

    it("should perform DIFFERENCE on animal sets", async () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source animals1: set<animal> = {
          {type: "dog", color: "red"},
          {type: "cat", color: "blue"}
        };

        source animals2: set<animal> = {
          {type: "dog", color: "red"}
        };

        transform unique_animals: set<animal> = DIFFERENCE(animals1, animals2);

        output result: set<animal> = unique_animals;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = await runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(1); // only the cat
    });
  });
});
