import { Compiler } from "./packages/compiler/dist/compiler.js";
import { Runtime } from "./packages/runtime/dist/runtime.js";

const compiler = new Compiler();
const runtime = new Runtime();

const source = `
source a: fraction = { kind: "fraction", numerator: 1, denominator: 2 };
source b: fraction = { kind: "fraction", numerator: 1, denominator: 4 };
transform result: fraction = ADD(a, b);
output final: fraction = result;
`;

const result = compiler.compile(source);
console.log("Success:", result.success);
console.log("Errors:", JSON.stringify(result.errors, null, 2));

if (result.success && result.program) {
  runtime.loadProgram(result.program);
  const outputs = runtime.execute(0);
  console.log("Output:", JSON.stringify(outputs, null, 2));
  console.log("Expected: { kind: 'fraction', numerator: 3, denominator: 4 }");
}
