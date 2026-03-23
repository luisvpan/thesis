import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Curriculum Types", () => {
  describe("Test 3.1: Filter Red Shapes", () => {
    it("should compile and execute shape filtering program", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source shapes: set<shape> = {
          {type: "circle", size: "large", color: "red"},
          {type: "triangle", size: "small", color: "blue"},
          {type: "square", size: "medium", color: "red"},
          {type: "circle", size: "large", color: "yellow"}
        };

        source target_color: text = "red";

        transform red_shapes: set<shape> = FILTER_BY_COLOR(shapes, target_color);

        output result: set<shape> = red_shapes;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      const result = outputs[0] as { kind: string; elements: unknown[] };
      expect(result.kind).toBe("set");
      expect(result.elements).toHaveLength(2); // 2 red shapes
    });

    it("should preserve immutability of sets", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source shapes: set<shape> = {
          {type: "circle", size: "large", color: "red"},
          {type: "triangle", size: "small", color: "blue"}
        };

        source target_color: text = "red";

        transform red_shapes: set<shape> = FILTER_BY_COLOR(shapes, target_color);

        output result: set<shape> = red_shapes;
      `;

      const compileResult = compiler.compile(source);
      runtime.loadProgram(compileResult.program!);
      runtime.execute();

      // Verify original set unchanged - source value is a wrapped set object
      const graph = runtime.getGraph();
      const sourceNode = graph.getNode("shapes");
      expect(sourceNode).toBeDefined();
      if (sourceNode && sourceNode.type === "DataSource") {
        const originalShapes = sourceNode.value as { kind: string; elements: unknown[] };
        expect(originalShapes.kind).toBe("set");
        expect(originalShapes.elements).toHaveLength(2);
      }
    });
  });
});
