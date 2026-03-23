import type { DataflowNode, DataflowEdge } from "@dataflow/shared/types";
import { OPERATION_REGISTRY } from "@dataflow/shared/operations";
import { DemandDrivenEvaluator } from "../evaluator";

export class DataflowGraph {
  private nodes = new Map<string, DataflowNode>();
  private edges: DataflowEdge[] = new Array<DataflowEdge>();
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
}

export class MissingInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MissingInputError";
  }
}
