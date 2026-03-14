import { Compiler } from "./packages/compiler/dist/compiler.js";

const compiler = new Compiler();

const source = `
  transform result: natural = ADD(10, 20);
  output final: natural = result;
`;

const result = compiler.compile(source);
console.log("Success:", result.success);
if (result.success && result.program) {
  console.log("Nodes:", JSON.stringify(result.program.graph.nodes, null, 2));
}
