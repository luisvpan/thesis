import { DataflowGraph } from "./graph/index.js";
import { DemandDrivenEvaluator } from "./evaluator/index.js";
import type { DataflowProgram } from "@dataflow/shared/types";

export class Runtime {
  private readonly MAX_MEMORY_MB = 100;
  private readonly MAX_MEMORY_BYTES = 100 * 1024 * 1024;
  private initialMemoryUsage: number;
  
  private graph: DataflowGraph;
  private evaluator: DemandDrivenEvaluator;
 
  constructor() {
    this.initialMemoryUsage = process.memoryUsage().heapUsed;
    this.graph = new DataflowGraph();
    this.evaluator = new DemandDrivenEvaluator();
  }

  loadProgram(program: DataflowProgram): void {
    if (!program || !program.graph) {
      throw new Error("Invalid program: program or program.graph is undefined");
    }

    this.graph = new DataflowGraph();
    this.evaluator = new DemandDrivenEvaluator();

    for (const node of program.graph.nodes) {
      this.graph.addNode(node);
    }

    for (const edge of program.graph.edges) {
      this.graph.addEdge(edge);
    }

    this.evaluator.clearCache();
  }

  getExecutionTrace(): { executionOrder: string[]; nodeEvaluations: Record<string, { value: unknown; timestep: number }> } {
    return this.evaluator.getExecutionTrace();
  }

  async execute(time: number = 0): Promise<unknown[]> {
    const beforeMemory = process.memoryUsage().heapUsed;
    const currentMemory = beforeMemory - this.initialMemoryUsage;

    if (currentMemory > this.MAX_MEMORY_BYTES) {
      throw new Error(`Memory limit exceeded: ${Math.round(currentMemory / 1024 / 1024)}MB used, limit is ${this.MAX_MEMORY_MB}MB`);
    }

    const outputs: unknown[] = [];

    for (const node of this.graph.getOutputNodes()) {
      const value = await this.evaluator.evaluate(node.id, time, this.graph);
      outputs.push(value);

      const afterMemory = process.memoryUsage().heapUsed;
      const memoryUsed = afterMemory - this.initialMemoryUsage;
      if (memoryUsed > this.MAX_MEMORY_BYTES) {
        throw new Error(`Memory limit exceeded during evaluation: ${Math.round(memoryUsed / 1024 / 1024)}MB used, limit is ${this.MAX_MEMORY_MB}MB`);
      }
    }

    return outputs;
  }

  getGraph(): DataflowGraph {
    return this.graph;
  }

  getEvaluator(): DemandDrivenEvaluator {
    return this.evaluator;
  }

  getCacheStats(): { hits: number; misses: number } {
    return this.evaluator.getCacheStats();
  }
}
