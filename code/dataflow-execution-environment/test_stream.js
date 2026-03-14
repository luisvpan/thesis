import { Compiler } from "./packages/compiler/dist/compiler.js";

const compiler = new Compiler();

const source = `
source numbers: stream<natural> = stream<natural>(generator(counter));
transform first_number: natural = FIRST(numbers);
output result: natural = first_number;
`;

const result = compiler.compile(source);
console.log("Success:", result.success);
console.log("Errors:", JSON.stringify(result.errors, null, 2));
