import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

function generateParallelizableProgram(nodeCount: number): string {
  const sourceCount = Math.floor(nodeCount * 0.5);
  const transformCount = nodeCount - sourceCount - 1;
  
  let source = "";
  
  for (let i = 0; i < sourceCount; i++) {
    source += `source src${i}: natural = ${i};\n`;
  }
  
  source += "\n";
  
  for (let i = 0; i < transformCount; i++) {
    const input1 = `src${i % sourceCount}`;
    const input2 = `src${(i + 1) % sourceCount}`;
    source += `transform t${i}: natural = ADD(${input1}, ${input2});\n`;
  }
  
  source += `\noutput result: natural = t${transformCount - 1};\n`;
  
  return source;
}

describe("Parallelism Benchmark", () => {
  it("should measure performance of 50-node program", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();
    
    const source = generateParallelizableProgram(50);
    
    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    
    runtime.loadProgram(compileResult.program!);
    
    const warmupRuns = 5;
    const measuredRuns = 10;
    
    for (let i = 0; i < warmupRuns; i++) {
      await runtime.execute();
    }
    
    const times: number[] = [];
    
    for (let i = 0; i < measuredRuns; i++) {
      const start = performance.now();
      await runtime.execute();
      const time = performance.now() - start;
      times.push(time);
    }
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    
    console.log(`Performance (sequential, ${measuredRuns} runs):`);
    console.log(`  Average: ${avgTime.toFixed(2)}ms`);
    console.log(`  Min: ${minTime.toFixed(2)}ms`);
    console.log(`  Max: ${maxTime.toFixed(2)}ms`);
    
    expect(avgTime).toBeLessThan(50);
  });
});
