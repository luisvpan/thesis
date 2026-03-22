import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Parallelism Benchmark - Before Optimization", () => {
  it("should measure baseline performance of program with multiple independent output nodes", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();
    
    const source = `
      source src0: natural = 10;
      source src1: natural = 20;
      source src2: natural = 30;
      source src3: natural = 40;
      source src4: natural = 50;
      source src5: natural = 60;
      source src6: natural = 70;
      source src7: natural = 80;
      source src8: natural = 90;
      source src9: natural = 100;
      
      transform t0: natural = ADD(src0, src1);
      transform t1: natural = ADD(src2, src3);
      transform t2: natural = ADD(src4, src5);
      transform t3: natural = ADD(src6, src7);
      transform t4: natural = ADD(src8, src9);
      
      transform t5: natural = MULTIPLY(t0, 2);
      transform t6: natural = MULTIPLY(t1, 2);
      transform t7: natural = MULTIPLY(t2, 2);
      transform t8: natural = MULTIPLY(t3, 2);
      transform t9: natural = MULTIPLY(t4, 2);
      
      transform t10: natural = ADD(t5, t6);
      transform t11: natural = ADD(t7, t8);
      transform t12: natural = ADD(t9, t5);
      
      output out0: natural = t10;
      output out1: natural = t11;
      output out2: natural = t12;
    `;
    
    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    
    runtime.loadProgram(compileResult.program!);
    
    const warmupRuns = 5;
    const measuredRuns = 20;
    
    for (let i = 0; i < warmupRuns; i++) {
      await runtime.execute();
    }
    
    const times: number[] = [];
    
    for (let i = 0; i < measuredRuns; i++) {
      const start = performance.now();
      const outputs = await runtime.execute();
      const time = performance.now() - start;
      times.push(time);
      
      expect(outputs).toHaveLength(3);
    }
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    
    console.log(`\nBaseline Performance (sequential evaluation):`);
    console.log(`  Runs: ${measuredRuns}`);
    console.log(`  Average: ${avgTime.toFixed(4)}ms`);
    console.log(`  Min: ${minTime.toFixed(4)}ms`);
    console.log(`  Max: ${maxTime.toFixed(4)}ms`);
    
    expect(avgTime).toBeLessThan(20);
  });

  it("should measure baseline performance of complex expression tree", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();
    
    const source = `
      source a: natural = 1;
      source b: natural = 2;
      source c: natural = 3;
      source d: natural = 4;
      source e: natural = 5;
      source f: natural = 6;
      source g: natural = 7;
      source h: natural = 8;
      
      transform t1: natural = ADD(a, b);
      transform t2: natural = ADD(c, d);
      transform t3: natural = ADD(e, f);
      transform t4: natural = ADD(g, h);
      
      transform t5: natural = MULTIPLY(t1, t2);
      transform t6: natural = MULTIPLY(t3, t4);
      
      transform t7: natural = ADD(t5, t6);
      
      output result: natural = t7;
    `;
    
    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    
    runtime.loadProgram(compileResult.program!);
    
    const warmupRuns = 5;
    const measuredRuns = 20;
    
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
    
    console.log(`\nBaseline Performance (complex expression tree):`);
    console.log(`  Runs: ${measuredRuns}`);
    console.log(`  Average: ${avgTime.toFixed(4)}ms`);
    console.log(`  Min: ${minTime.toFixed(4)}ms`);
    console.log(`  Max: ${maxTime.toFixed(4)}ms`);
    
    expect(avgTime).toBeLessThan(10);
  });
});
