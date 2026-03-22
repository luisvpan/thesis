import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Parallelism Verification", () => {
  it("should execute program with multiple independent outputs using parallel evaluation", async () => {
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
    const measuredRuns = 50;
    
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
    
    console.log(`\nPerformance with async parallel evaluation (${measuredRuns} runs):`);
    console.log(`  Average: ${avgTime.toFixed(4)}ms`);
    console.log(`  Min: ${minTime.toFixed(4)}ms`);
    console.log(`  Max: ${maxTime.toFixed(4)}ms`);
    
    expect(avgTime).toBeLessThan(10);
  });

  it("should demonstrate parallel evaluation benefits on wide programs", async () => {
    const compiler = new Compiler();
    const runtime = new Runtime();
    
    const source = `
      source src0: natural = 1;
      source src1: natural = 2;
      source src2: natural = 3;
      source src3: natural = 4;
      source src4: natural = 5;
      source src5: natural = 6;
      source src6: natural = 7;
      source src7: natural = 8;
      source src8: natural = 9;
      source src9: natural = 10;
      
      transform t0: natural = MULTIPLY(src0, 2);
      transform t1: natural = MULTIPLY(src1, 2);
      transform t2: natural = MULTIPLY(src2, 2);
      transform t3: natural = MULTIPLY(src3, 2);
      transform t4: natural = MULTIPLY(src4, 2);
      transform t5: natural = MULTIPLY(src5, 2);
      transform t6: natural = MULTIPLY(src6, 2);
      transform t7: natural = MULTIPLY(src7, 2);
      transform t8: natural = MULTIPLY(src8, 2);
      transform t9: natural = MULTIPLY(src9, 2);
      
      output out0: natural = t0;
      output out1: natural = t1;
      output out2: natural = t2;
      output out3: natural = t3;
      output out4: natural = t4;
      output out5: natural = t5;
      output out6: natural = t6;
      output out7: natural = t7;
      output out8: natural = t8;
      output out9: natural = t9;
    `;
    
    const compileResult = compiler.compile(source);
    expect(compileResult.success).toBe(true);
    
    runtime.loadProgram(compileResult.program!);
    
    const warmupRuns = 5;
    const measuredRuns = 50;
    
    for (let i = 0; i < warmupRuns; i++) {
      await runtime.execute();
    }
    
    const times: number[] = [];
    
    for (let i = 0; i < measuredRuns; i++) {
      const start = performance.now();
      const outputs = await runtime.execute();
      const time = performance.now() - start;
      times.push(time);
      
      expect(outputs).toHaveLength(10);
    }
    
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    
    console.log(`\nPerformance on 10 independent outputs (${measuredRuns} runs):`);
    console.log(`  Average: ${avgTime.toFixed(4)}ms`);
    console.log(`  Min: ${minTime.toFixed(4)}ms`);
    console.log(`  Max: ${maxTime.toFixed(4)}ms`);
    
    expect(avgTime).toBeLessThan(10);
  });
});
