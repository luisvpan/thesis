import { Compiler } from './packages/compiler/dist/compiler.js';

const compiler = new Compiler();

const source = 'source a: natural = 5;\noutput result: natural = a;';
const result = compiler.compile(source);
console.log('Success:', result.success);
console.log('Has program:', !!result.program);
console.log('Nodes:', result.program?.graph.nodes.length);
console.log('Edges:', result.program?.graph.edges.length);
console.log('\nProgram structure:', JSON.stringify(result.program, null, 2));

const source2 = 'source a: natural = 3;\nsource b: natural = 2;\ntransform sum: natural = ADD(a, b);\noutput result: natural = sum;';
const result2 = compiler.compile(source2);
console.log('\n\n=== Test 2 ===');
console.log('Success:', result2.success);
console.log('Nodes:', result2.program?.graph.nodes.length);
console.log('Edges:', result2.program?.graph.edges.length);

const source3 = 'source a: natural = 3;\nsource a: natural = 2;';
const result3 = compiler.compile(source3);
console.log('\n\n=== Test 3 (duplicate) ===');
console.log('Success:', result3.success);
console.log('Errors:', result3.errors.length);
console.log('Error code:', result3.errors[0]?.code);
console.log('Child message:', result3.errors[0]?.childMessage);
