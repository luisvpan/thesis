import { Compiler } from "./packages/compiler/dist/compiler.js";
import { Runtime } from "./packages/runtime/dist/runtime.js";

const compiler = new Compiler();

const source = `
  source a: natural = 5;
  transform result: natural = ADD(a, 10);
  output final: natural = result;
`;

const result = compiler.compile(source);
console.log("Success:", result.success);
if (result.success && result.program) {
  console.log("Nodes:", JSON.stringify(result.program.graph.nodes, null, 2));
}
