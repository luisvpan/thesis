import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";

describe("Compiler - Inline Nested Operations", () => {
  it("should parse AST for inline nested operations", () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 1;

      transform final_sum: natural = ADD(ADD(a, b), c);

      output result: natural = final_sum;
    `;

    const ast = compiler.parse(source);
    
    console.log("\n=== AST Structure ===");
    console.log(JSON.stringify(ast, null, 2));
  });

  it("should compile program with inline nested operations", () => {
    const compiler = new Compiler();
    const source = `
      source a: natural = 3;
      source b: natural = 2;
      source c: natural = 1;

      transform final_sum: natural = ADD(ADD(a, b), c);

      output result: natural = final_sum;
    `;

    const result = compiler.compile(source);
    
    console.log("\n=== Compilation Result ===");
    console.log("Success:", result.success);
    if (!result.success) {
      console.log("Errors:", JSON.stringify(result.errors, null, 2));
    }
    
    if (result.success && result.program) {
      console.log("\n=== Graph Nodes ===");
      console.log(JSON.stringify(result.program.graph.nodes, null, 2));
      console.log("\n=== Graph Edges ===");
      console.log(JSON.stringify(result.program.graph.edges, null, 2));
    }
  });
});
