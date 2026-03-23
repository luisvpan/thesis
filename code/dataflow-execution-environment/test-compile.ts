import { Compiler } from "@dataflow/compiler";

const compiler = new Compiler();
const source = `source numbers: set<decimal> = {1.5, 2.5, 0.5, 3.5, 4.5};
transform filtered: set<decimal> = FILTER(numbers, 3.5);
output result: set<decimal> = filtered;`;

const compileResult = compiler.compile(source);
console.log("Success:", compileResult.success);
if (!compileResult.success) {
  console.log("Errors:", JSON.stringify(compileResult.errors, null, 2));
} else {
  console.log("Success!");
}
