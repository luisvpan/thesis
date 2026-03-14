import { Compiler } from "./packages/compiler/dist/compiler.js";

const compiler = new Compiler();
const source = `source a: natural = 3;
source b: natural = 2;
source c: natural = 1;
transform result: natural = ADD(ADD(a, b), c);
output final: natural = result;`;

const result = compiler.compile(source);
console.log("Success:", result.success);
if (!result.success && result.errors.length > 0) {
  console.log("Error Code:", result.errors[0].code);
  console.log("Error Message:", result.errors[0].message);
  console.log("Node ID:", result.errors[0].nodeId);
  console.log("Full Error:", JSON.stringify(result.errors[0], null, 2));
} else {
  console.log("No errors - SUCCESS!");
  console.log("Program:", result.program ? "Created" : "None");
}
