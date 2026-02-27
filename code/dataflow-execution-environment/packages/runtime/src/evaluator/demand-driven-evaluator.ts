import { DataflowGraph } from "../graph/dataflow-graph.js";
import { ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE } from "../operations/numeric.js";

export class DemandDrivenEvaluator {
  private cache = new Map<string, Map<number, unknown>>();
  private cacheHits = 0;
  private cacheMisses = 0;

  evaluate(nodeId: string, time: number, graph: DataflowGraph): unknown {
    if (this.cache.has(nodeId) && this.cache.get(nodeId)!.has(time)) {
      this.cacheHits++;
      return this.cache.get(nodeId)!.get(time);
    }
    this.cacheMisses++;

    const node = graph.getNode(nodeId);
    if (!node) throw new Error(`Node ${nodeId} not found`);

    let value: unknown;
    if (node.type === "DataSource") {
      value = node.value;
    } else if (node.type === "Transformation") {
      const inputNodes = graph.getInputs(nodeId);
      value = this.evaluateOperation(node.operation, inputNodes, time, graph);
    } else if (node.type === "Output") {
      const inputNodes = graph.getInputs(nodeId);
      if (inputNodes.length === 0) {
        throw new Error(`Output node ${nodeId} has no input`);
      }
      value = this.evaluate(inputNodes[0].id, time, graph);
    }

    if (!this.cache.has(nodeId)) {
      this.cache.set(nodeId, new Map());
    }
    this.cache.get(nodeId)!.set(time, value);

    return value;
  }

  private evaluateOperation(operation: string, inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph): unknown {
    const evaluatedInputs = inputs.map(input => ({
      id: input.id,
      value: this.evaluate(input.id, time, graph)
    }));

    switch (operation) {
      case "ADD":
        return ADD(evaluatedInputs);
      case "SUBTRACT":
        return SUBTRACT(evaluatedInputs);
      case "MULTIPLY":
        return MULTIPLY(evaluatedInputs);
      case "DIVIDE":
        return DIVIDE(evaluatedInputs);
      case "COMPARE":
        return COMPARE(evaluatedInputs);
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
  }

  getCacheStats(): { hits: number; misses: number } {
    return {
      hits: this.cacheHits,
      misses: this.cacheMisses
    };
  }

  clearCache(): void {
    this.cache.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
  }
}
