import { Compiler } from "./compiler.js";

const compiler = new Compiler();
const source = `
  source a: natural = 5;
  source b: natural = 3;
  source c: natural = 10;
  transform sum: natural = ADD(a, b);
  transform unused: natural = MULTIPLY(c, 2);
  output result: natural = sum;
`;

const result = compiler.compile(source);
console.log('Success:', result.success);
if (!result.success) {
  console.log('Errors:', JSON.stringify(result.errors, null, 2));
} else {
  console.log('Nodes:', result.program.graph.nodes.length);
  console.log('Node IDs:', result.program.graph.nodes.map(n => n.id).join(', '));
}
