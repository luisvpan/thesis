import { Compiler } from './packages/compiler/dist/compiler.js';

const compiler = new Compiler();

console.log('Test 1 - Simple:');
const source1 = 'source a: natural = 5;\noutput result: natural = a;';
const result1 = compiler.compile(source1);
console.log('  Success:', result1.success);
console.log('  Has program:', !!result1.program);
console.log('  Nodes:', result1.program?.graph.nodes.length);
console.log('  Edges:', result1.program?.graph.edges.length);

console.log('\nTest 2 - Multiple statements:');
const source2 = 'source a: natural = 3;\nsource b: natural = 2;\ntransform sum: natural = ADD(a, b);\noutput result: natural = sum;';
const result2 = compiler.compile(source2);
console.log('  Success:', result2.success);
console.log('  Has program:', !!result2.program);
console.log('  Nodes:', result2.program?.graph.nodes.length);
console.log('  Edges:', result2.program?.graph.edges.length);

console.log('\nTest 3 - Duplicate (should fail):');
const source3 = 'source a: natural = 3;\nsource a: natural = 2;';
const result3 = compiler.compile(source3);
console.log('  Success:', result3.success);
console.log('  Errors:', result3.errors.length);
console.log('  Error code:', result3.errors[0]?.code);
