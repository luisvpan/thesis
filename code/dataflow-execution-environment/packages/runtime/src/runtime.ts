import { DataflowGraph } from "./graph/index.js";
import { DemandDrivenEvaluator } from "./evaluator/index.js";
import type { DataflowProgram } from "@dataflow/shared/types";

export class Runtime {
  private graph: DataflowGraph;
  private evaluator: DemandDrivenEvaluator;

  constructor() {
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
  }

  execute(time: number = 0): unknown[] {
    const outputs: unknown[] = [];

    for (const node of this.graph.getOutputNodes()) {
      const value = this.evaluator.evaluate(node.id, time, this.graph);
      outputs.push(value);
    }

    return outputs;
  }

  getGraph(): DataflowGraph {
    return this.graph;
  }

  getEvaluator(): DemandDrivenEvaluator {
    return this.evaluator;
  }
}
