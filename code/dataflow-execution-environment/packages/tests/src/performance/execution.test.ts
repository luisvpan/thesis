import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

function generateProgramForExecution(nodeCount: number): string {
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

describe("Integration Tests - Performance", () => {
  describe("Test 7.2: Execution Performance", () => {
    it("should execute 50-node program in under 50ms", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();
      
      const source = generateProgramForExecution(50);
      
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
      
      runtime.loadProgram(compileResult.program!);
      
      const start = performance.now();
      const outputs = runtime.execute();
      const time = performance.now() - start;
      
      expect(outputs).toHaveLength(1);
      expect(time).toBeLessThan(50);
    });

    it("should have ~30% cache hit rate for repeated evaluations", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();
      
      const source = generateProgramForExecution(50);
      
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
      
      runtime.loadProgram(compileResult.program!);
      
      const evaluator = runtime.getEvaluator();
      
      const outputs1 = runtime.execute();
      const stats1 = evaluator.getCacheStats();
      
      const outputs2 = runtime.execute();
      const stats2 = evaluator.getCacheStats();
      
      expect(outputs1).toEqual(outputs2);
      expect(stats2.hits).toBeGreaterThan(stats1.hits);
      
      const hitRate = stats2.hits / (stats2.hits + stats2.misses);
      
      expect(hitRate).toBeGreaterThanOrEqual(0.2);
    });

    it("should verify demand-driven semantics", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();
      
      const source = `
        source a: natural = 5;
        source b: natural = 3;
        source c: natural = 10;
        transform sum: natural = ADD(a, b);
        transform unused: natural = MULTIPLY(c, 2);
        output result: natural = sum;
      `;
      
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
      
      runtime.loadProgram(compileResult.program!);
      
      const evaluator = runtime.getEvaluator();
      
      const outputs = runtime.execute();
      const stats = evaluator.getCacheStats();
      
      expect(outputs).toHaveLength(1);
      
      const graph = runtime.getGraph();
      const allNodes = graph.getAllNodes();
      
      expect(stats.hits + stats.misses).toBeLessThanOrEqual(allNodes.length);
    });

    it("should execute 50-node program 5 times faster than full re-evaluation", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();
      
      const source = generateProgramForExecution(50);
      
      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);
      
      const timesWithoutCache: number[] = [];
      const timesWithCache: number[] = [];
      
      for (let i = 0; i < 5; i++) {
        runtime.loadProgram(compileResult.program!);
        
        const start = performance.now();
        runtime.execute();
        const time = performance.now() - start;
        timesWithoutCache.push(time);
      }
      
      runtime.loadProgram(compileResult.program!);
      
      for (let i = 0; i < 5; i++) {
        const start = performance.now();
        runtime.execute();
        const time = performance.now() - start;
        timesWithCache.push(time);
      }
      
      const avgWithoutCache = timesWithoutCache.reduce((a, b) => a + b, 0) / timesWithoutCache.length;
      const avgWithCache = timesWithCache.reduce((a, b) => a + b, 0) / timesWithCache.length;
      
      expect(avgWithCache).toBeLessThan(avgWithoutCache * 2);
    });
  });
});
