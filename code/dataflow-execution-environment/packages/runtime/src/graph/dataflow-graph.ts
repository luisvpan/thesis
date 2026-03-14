import type { DataflowNode, DataflowEdge } from "@dataflow/shared/types";
import { OPERATION_REGISTRY } from "@dataflow/shared/operations";

export class DataflowGraph {
  private nodes = new Map<string, DataflowNode>();
  private edges = DataflowEdge[];
  private cache = new Map<string, unknown>();

  addNode(node: DataflowNode): void {
    this.nodes.set(node.id, node);
  }

  addEdge(edge: DataflowEdge): void {
    this.edges.push(edge);
  }

  getNode(nodeId: string): DataflowNode | undefined {
    return this.nodes.get(nodeId);
  }

  getInputs(nodeId: string): Array<{ id: string; value: unknown }> {
    const incoming = this.edges.filter(e => e.to === nodeId).sort((a, b) => (a.toPort ?? 0) - (b.toPort ?? 0));
    const result: Array<{ id: string; value: unknown }> = [];

    for (const edge of incoming) {
      const node = this.nodes.get(edge.from);
      if (!node) {
        throw new Error(`Node '${edge.from}' not found`);
      }
      result.push({ id: node.id, value: (node as any).value });
    }

    const node = this.nodes.get(nodeId);
    if (node && node.type === "Transformation") {
      const signature = (OPERATION_REGISTRY as any)[node.operation];
      if (signature && result.length < signature.arity) {
        throw new Error(`Missing input at port ${result.length} for ${node.operation}`);
      }
    }

    return result;
  }

  getOutputNodes(): DataflowNode[] {
    return Array.from(this.nodes.values()).filter(n => n.type === "Output");
  }

  getAllNodes(): DataflowNode[] {
    return Array.from(this.nodes.values());
  }

  getAllEdges(): DataflowEdge[] {
    return [...this.edges];
  }

  evaluate(nodeId: string, time: number, graph: DataflowGraph): unknown {
    const cacheKey = `${nodeId}_${time}`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }

    let result: unknown;

    if (node.type === "DataSource") {
      result = node.value;
    } else if (node.type === "Transformation") {
      const inputNodes = this.getInputs(nodeId);
      const { DemandDrivenEvaluator } = require("../evaluator/index.js");
      const evaluator = new DemandDrivenEvaluator();
      result = evaluator.evaluateOperation(node.operation, inputNodes, time, graph);
    } else if (node.type === "Output") {
      const inputNodes = this.getInputs(nodeId);
      if (inputNodes.length > 0) {
        const { DemandDrivenEvaluator } = require("../evaluator/index.js");
        const outputEvaluator = new DemandDrivenEvaluator();
        result = outputEvaluator.evaluate(inputNodes[0].id, time, graph);
      } else {
        result = undefined;
      }
    } else {
      throw new Error(`Unknown node type: ${(node as any).type}`);
    }

    this.cache.set(cacheKey, result);
    return result;
  }
}

export class MissingInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MissingInputError";
  }
}
