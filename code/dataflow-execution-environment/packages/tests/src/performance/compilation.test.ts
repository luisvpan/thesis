import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

function generateLargeProgram(nodeCount: number): string {
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
  describe("Test 7.1: Compilation Performance", () => {
    it("should compile 100-node program in under 100ms", () => {
      const compiler = new Compiler();
      
      const source = generateLargeProgram(100);
      
      const start = performance.now();
      const result = compiler.compile(source);
      const time = performance.now() - start;
      
      expect(result.success).toBe(true);
      expect(time).toBeLessThan(100);
    });

    it("should compile 100-node program with linear scaling", () => {
      const compiler = new Compiler();
      
      const times: number[] = [];
      const nodeCounts = [10, 25, 50, 100];
      
      for (const count of nodeCounts) {
        const source = generateLargeProgram(count);
        
        const start = performance.now();
        const result = compiler.compile(source);
        const time = performance.now() - start;
        
        expect(result.success).toBe(true);
        times.push(time);
      }
      
      for (let i = 1; i < times.length; i++) {
        const ratio = times[i] / times[i - 1];
        const expectedRatio = nodeCounts[i] / nodeCounts[i - 1];

        expect(ratio).toBeLessThan(expectedRatio * 1.6);
      }
    });

    it("should have no memory leaks across 100 compilations", () => {
      const compiler = new Compiler();
      
      const source = generateLargeProgram(100);
      const initialMemory = process.memoryUsage().heapUsed;
      
      for (let i = 0; i < 100; i++) {
        const result = compiler.compile(source);
        expect(result.success).toBe(true);
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
    });
  });
});
