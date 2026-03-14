import { Compiler } from "./packages/compiler/dist/compiler.js";
import { Runtime } from "./packages/runtime/dist/runtime.js";

const compiler = new Compiler();
const runtime = new Runtime();

const source1 = `source a: natural = 3;
source b: natural = 2;
source c: natural = 1;
transform result: natural = ADD(ADD(a, b), c);
output final: natural = result;`;

console.log("=== Test 1: ADD(ADD(a, b), c) ===");
const result1 = compiler.compile(source1);
console.log("Success:", result1.success);
if (result1.success) {
  runtime.loadProgram(result1.program);
  const outputs = runtime.execute(0);
  console.log("Output:", outputs);
  console.log("Expected: [{kind:'natural',value:6}]");
}

console.log("\n=== Test 2: MULTIPLY(ADD(a, b), SUBTRACT(c, d)) ===");
const source2 = `source a: natural = 3;
source b: natural = 2;
source c: natural = 10;
source d: natural = 6;
transform result: natural = MULTIPLY(ADD(a, b), SUBTRACT(c, d));
output final: natural = result;`;

const result2 = compiler.compile(source2);
console.log("Success:", result2.success);
if (result2.success) {
  runtime.loadProgram(result2.program);
  const outputs = runtime.execute(0);
  console.log("Output:", outputs);
  console.log("Expected: [{kind:'natural',value:20}]");
}

console.log("\n=== Test 3: Deep nesting ADD(ADD(ADD(a, b), c), d) ===");
const source3 = `source a: natural = 1;
source b: natural = 2;
source c: natural = 3;
source d: natural = 4;
transform result: natural = ADD(ADD(ADD(a, b), c), d);
output final: natural = result;`;

const result3 = compiler.compile(source3);
console.log("Success:", result3.success);
if (result3.success) {
  runtime.loadProgram(result3.program);
  const outputs = runtime.execute(0);
  console.log("Output:", outputs);
  console.log("Expected: [{kind:'natural',value:10}]");
}

console.log("\n=== Test 4: Type error in nested operations ===");
const source4 = `source a: natural = 5;
source b: text = "hello";
source c: natural = 3;
transform result: natural = ADD(a, b);
output final: natural = result;`;

const result4 = compiler.compile(source4);
console.log("Success:", result4.success);
if (!result4.success) {
  console.log("Error Code:", result4.errors[0].code);
  console.log("Expected: TYPE_ERROR");
}
