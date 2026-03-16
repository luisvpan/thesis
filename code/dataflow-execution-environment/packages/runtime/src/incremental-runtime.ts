import type { DataflowNode, DataflowEdge, DataflowProgram, DataType } from "@dataflow/shared/types";
import { OPERATION_REGISTRY } from "@dataflow/shared/operations";
import {
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  COMPARE
} from "./operations/numeric.js";
import { AND, OR, NOT } from "./operations/boolean.js";
import {
  COMPARE_BY_SIZE,
  COMPARE_BY_COLOR,
  COMPARE_BY_TYPE,
  COMPARE_BY_TASTE,
  COMPARE_BY_AGE_GROUP,
  COMPARE_BY_GENDER
} from "./operations/comparison.js";
import {
  FILTER,
  FILTER_BY_SIZE,
  FILTER_BY_COLOR,
  FILTER_BY_TYPE,
  FILTER_BY_TASTE,
  FILTER_BY_AGE_GROUP,
  FILTER_BY_GENDER
} from "./operations/filtering.js";
import {
  UNION,
  INTERSECTION,
  DIFFERENCE,
  COMPLEMENT,
  SORT,
  ALPHABETICAL_SORT
} from "./operations/sets.js";

export type NodeState =
  | { status: "completed"; value: any }
  | { status: "pending"; missingInputs: MissingInput[] }
  | { status: "processing" }
  | { status: "error"; error: string };

export type MissingInput = {
  port: number;
  nodeId: string;
  description: string;
  childMessage: string;
};

export type NodeEvaluationResult = {
  nodeId: string;
  state: NodeState;
  value?: any;
  missingInputs?: MissingInput[];
  error?: string;
};

export type PartialEvaluation = {
  nodeStates: Map<string, NodeState>;
  changedNodes: string[];
};

export type GraphDelta = {
  addedNodes?: DataflowNode[];
  removedNodes?: string[];
  addedEdges?: DataflowEdge[];
  removedEdges?: DataflowEdge[];
};

export type UpdateResult = {
  changedNodes: string[];
  errors: string[];
};

export type Subscriber = {
  nodeId: string;
  callback: (state: NodeState) => void;
};

type StateCallback = (state: NodeState) => void;

export class IncrementalRuntime {
  private graph: Map<string, DataflowNode>;
  private edges: Map<string, DataflowEdge>;
  private cache: Map<string, Map<number, unknown>>;
  private subscriptions: Map<string, Set<StateCallback>>;
  private nodeDependencies: Map<string, Set<string>>;

  constructor() {
    this.graph = new Map();
    this.edges = new Map();
    this.cache = new Map();
    this.subscriptions = new Map();
    this.nodeDependencies = new Map();
  }

  loadProgram(program: DataflowProgram): void {
    for (const node of program.graph.nodes) {
      this.graph.set(node.id, node);
    }
    for (const edge of program.graph.edges) {
      this.edges.set(edge.id, edge);
    }
    this.cache.clear();
    this.buildDependencyGraph();
  }

  updateGraph(delta: GraphDelta): UpdateResult {
    const changedNodes: string[] = [];
    const errors: string[] = [];

    for (const node of delta.addedNodes || []) {
      this.graph.set(node.id, node);
      this.invalidateDependentCache(node.id);
      changedNodes.push(node.id);
      this.buildDependencyGraph();
    }

    for (const nodeId of delta.removedNodes || []) {
      this.graph.delete(nodeId);
      this.cache.delete(nodeId);
      this.subscriptions.delete(nodeId);
      changedNodes.push(nodeId);
    }

    for (const edge of delta.addedEdges || []) {
      this.edges.set(edge.id, edge);
    }

    for (const edge of delta.removedEdges || []) {
      this.edges.delete(edge.id);
    }

    if (changedNodes.length > 0) {
      this.notifySubscribers(changedNodes);
    }

    return {
      changedNodes,
      errors
    };
  }

  evaluatePartial(timestep: number = 0): PartialEvaluation {
    const nodeStates: Map<string, NodeState> = new Map();
    const changedNodes: string[] = [];

    const demandSources = this.findDemandSources();
    if (demandSources.length === 0) {
      return { nodeStates, changedNodes };
    }

    for (const source of demandSources) {
      try {
        const result = this.evaluateNode(source.id, timestep);
        nodeStates.set(source.id, result.state);

        if (result.state.status === "completed" && result.changedNodes) {
          changedNodes.push(...result.changedNodes);
        }
      } catch (e) {
        nodeStates.set(source.id, { status: "error", error: (e as Error).message });
      }
    }

    if (changedNodes.length > 0) {
      this.notifySubscribers(changedNodes);
    }

    return { nodeStates, changedNodes };
  }

  getNodeState(nodeId: string): NodeState {
    const cachedState = this.getCachedState(nodeId, 0);

    if (cachedState) {
      return cachedState;
    }

    try {
      const result = this.evaluateNode(nodeId, 0);
      return result.state;
    } catch (e) {
      return { status: "error", error: (e as Error).message };
    }
  }

  subscribe(nodeId: string, callback: StateCallback): void {
    if (!this.subscriptions.has(nodeId)) {
      this.subscriptions.set(nodeId, new Set());
    }

    const subscribers = this.subscriptions.get(nodeId)!;
    subscribers.add(callback);
  }

  unsubscribe(nodeId: string, callback: StateCallback): void {
    const subscribers = this.subscriptions.get(nodeId);
    if (subscribers) {
      subscribers.delete(callback);
      if (subscribers.size === 0) {
        this.subscriptions.delete(nodeId);
      }
    }
  }

  private findDemandSources(): DataflowNode[] {
    const outputNodes = Array.from(this.graph.values()).filter(
      n => n.type === "Output" || this.subscriptions.has(n.id)
    );

    if (outputNodes.length === 0) {
      return Array.from(this.subscriptions.keys())
        .map(id => this.graph.get(id)!)
        .filter(n => n !== undefined);
    }

    return outputNodes;
  }

  private evaluateNode(nodeId: string, time: number): { state: NodeState; value?: any; changedNodes: string[] } {
    const timeMap = this.cache.get(nodeId);
    if (timeMap) {
      const cached = timeMap.get(time);
      if (cached !== undefined) {
        return { state: { status: "completed", value: cached }, changedNodes: [] };
      }
    }

    const node = this.graph.get(nodeId);
    if (!node) {
      return { state: { status: "error", error: `Node ${nodeId} not found` }, changedNodes: [] };
    }

    try {
      if (node.type === "DataSource") {
        const value = node.value;
        const wrappedValue = this.wrapDataSourceValue(node, time);
        this.cache.set(nodeId, new Map([[time, wrappedValue]]));
        return { state: { status: "completed", value: wrappedValue }, changedNodes: [] };
      }

      if (node.type === "Output") {
        const inputEdges = Array.from(this.edges.values()).filter(e => e.to === nodeId);
        const inputNodes = inputEdges.map(e => e.from);
        if (inputNodes.length === 0) {
          return { state: { status: "pending", missingInputs: [{ port: 0, nodeId: "", description: "output node needs input", childMessage: "⚠️ ¡Ups! El bloque de salida necesita una entrada" }] }, changedNodes: [] };
        }

        const sourceNode = this.graph.get(inputNodes[0]);
        if (!sourceNode) {
          return { state: { status: "pending", missingInputs: [{ port: 0, nodeId: inputNodes[0], description: "input not found", childMessage: `⚠️ ¡Ups! No encuentro el bloque "${inputNodes[0]}".` }] }, changedNodes: [] };
        }

        const sourceResult = this.evaluateNode(sourceNode.id, time);
        if (sourceResult.state.status === "completed") {
          this.cache.set(nodeId, new Map([[time, (sourceResult.state as any).value]]));
          return { state: { status: "completed", value: sourceResult.state.value }, changedNodes: sourceResult.changedNodes };
        } else if (sourceResult.state.status === "pending") {
          return sourceResult;
        } else {
          return sourceResult;
        }
      }

      if (node.type === "Transformation") {
        const inputEdges = Array.from(this.edges.values()).filter(e => e.to === nodeId);
        const inputNodes = inputEdges.map(e => e.from);
        if (inputNodes.length === 0) {
          const signature = OPERATION_REGISTRY[node.operation as any];
          const inputCount = signature?.contracts[0]?.arity || 0;
          const missingInputs = this.createMissingInputs(node, inputCount);
          return { state: { status: "pending", missingInputs }, changedNodes: [] };
        }

        const evaluatedInputs = new Map<string, any>();
        const missingInputsList: MissingInput[] = [];
        let allInputsComplete = true;

        for (let i = 0; i < inputNodes.length; i++) {
          const inputNode = this.graph.get(inputNodes[i]);
          if (!inputNode) {
            allInputsComplete = false;
            missingInputsList.push({
              port: i,
              nodeId: inputNodes[i],
              description: this.getInputDescription(node.operation, i),
              childMessage: this.getChildMessageForMissingInput(node.operation, i)
            });
            continue;
          }

          try {
            const inputValue = this.evaluateNode(inputNode.id, time);
            evaluatedInputs.set(inputNodes[i], inputValue);
          } catch (e) {
            allInputsComplete = false;
            missingInputsList.push({
              port: i,
              nodeId: inputNodes[i],
              description: `Error: ${e}`,
              childMessage: `⚠️ ¡Ups! Hubo un error evaluando "${inputNodes[i]}".`
            });
          }
        }

        if (missingInputsList.length > 0 || !allInputsComplete) {
          return { state: { status: "pending", missingInputs: missingInputsList }, changedNodes: [] };
        }

        const signature = OPERATION_REGISTRY[node.operation as any];
        const contract = signature?.contracts[0];
        const evaluatedInputsArray = Array.from(evaluatedInputs.values()).map(v => v.state.status === 'completed' ? v.state.value : v.value);

        if (contract) {
          const result = this.executeOperation(node.operation, evaluatedInputsArray);
          this.cache.set(nodeId, new Map([[time, result]]));
          return { state: { status: "completed", value: result }, changedNodes: [] };
        } else {
          throw new Error(`Unknown operation: ${node.operation}`);
        }
      }
    } catch (e) {
      return { state: { status: "error", error: (e as Error).message }, changedNodes: [] };
    }

    return { state: { status: "error", error: "Unknown node type: ${node.type}" }, changedNodes: [] };
  }

  private wrapDataSourceValue(node: { type: "DataSource"; dataType: string | { kind: string; elementType?: any }; value: unknown }, time: number): any {
    const numValue = typeof node.value === "number" ? node.value : node.value;
    const dataType = typeof node.dataType === "string" ? node.dataType : (node.dataType as { kind: string; elementType?: any }).kind;

    switch (dataType) {
      case "natural":
        return { kind: "natural" as const, value: numValue as number };
      case "integer":
        return { kind: "integer" as const, value: numValue as number };
      case "decimal":
        return { kind: "decimal" as const, value: numValue as number };
      case "fraction":
        return { kind: "fraction" as const, numerator: (numValue as { numerator: number; denominator: number }).numerator, denominator: (numValue as { numerator: number; denominator: number }).denominator };
      case "text":
        return { kind: "text" as const, value: numValue as string };
      case "boolean":
        return { kind: "boolean" as const, value: numValue as boolean };
      case "shape":
        return { kind: "shape" as const, ...numValue as { type: string; size: string; color: string } };
      case "car":
        return { kind: "car" as const, ...numValue as { color: string } };
      case "food":
        return { kind: "food" as const, ...numValue as { taste: string; color: string } };
      case "animal":
        return { kind: "animal" as const, ...numValue as { type: string; color: string } };
      case "person":
        return { kind: "person" as const, ...numValue as { ageGroup: string; gender: string } };
      default:
        if (dataType.startsWith("stream")) {
          const streamValue = node.value as { kind: "stream"; elementType: string; generatorFactory?: () => Generator<unknown>; generator: Generator<unknown> };
          const gen = streamValue.generatorFactory ? streamValue.generatorFactory() : streamValue.generator;
          const firstValue = gen.next().value;
          return {
            kind: "stream" as const,
            elementType: streamValue.elementType,
            generator: gen,
            firstValue
          };
        }
        if (dataType.startsWith("set")) {
          const elementType = this.extractElementType(dataType);
          const elements = (numValue as unknown[]).map((elem: unknown) => {
            if (typeof elem === 'object' && elem !== null) {
              const record = elem as Record<string, unknown>;
              if (elementType === "shape") {
                return { kind: "shape", ...record };
              } else if (elementType === "car") {
                return { kind: "car", ...record };
              } else if (elementType === "food") {
                return { kind: "food", ...record };
              } else if (elementType === "animal") {
                return { kind: "animal", ...record };
              } else if (elementType === "person") {
                return { kind: "person", ...record };
              }
              return record;
            }
          });
          return { kind: "set", elements };
        }
        return node.value;
    }
  }

  private extractElementType(compositeType: string): string {
    const match = compositeType.match(/^(set|stream)<(.+)>$/);
    return match ? match[2] : "unknown";
  }

  private executeOperation(operation: string, inputs: unknown[]): unknown {
    const signature = OPERATION_REGISTRY[operation as any];
    if (!signature) {
      throw new Error(`Unknown operation: ${operation}`);
    }

    const contract = signature.contracts.find(c => {
      return c.inputTypes.every((inputType, i) => {
        const input = inputs[i];
        if (input === null || input === undefined) return false;

        if (typeof inputType === 'string') {
          return (input as any).kind === inputType;
        }

        if (typeof inputType === 'object') {
          return (input as any).kind === (inputType as any).kind;
        }

        return false;
      });
    });

    if (!contract) {
      throw new Error(`No contract found for operation ${operation} with inputs ${inputs.map(i => (i as any).kind).join(', ')}`);
    }

    const wrappedInputs = inputs.map((value, index) => ({ id: `input_${index}`, value }));

    switch (operation) {
      case "ADD":
        return ADD(wrappedInputs);
      case "SUBTRACT":
        return SUBTRACT(wrappedInputs);
      case "MULTIPLY":
        return MULTIPLY(wrappedInputs);
      case "DIVIDE":
        return DIVIDE(wrappedInputs);
      case "COMPARE":
        return COMPARE(wrappedInputs);
      case "AND":
        return AND(wrappedInputs);
      case "OR":
        return OR(wrappedInputs);
      case "NOT":
        return NOT(wrappedInputs);
      case "COMPARE_BY_SIZE":
        return COMPARE_BY_SIZE(wrappedInputs);
      case "COMPARE_BY_COLOR":
        return COMPARE_BY_COLOR(wrappedInputs);
      case "COMPARE_BY_TYPE":
        return COMPARE_BY_TYPE(wrappedInputs);
      case "COMPARE_BY_TASTE":
        return COMPARE_BY_TASTE(wrappedInputs);
      case "COMPARE_BY_AGE_GROUP":
        return COMPARE_BY_AGE_GROUP(wrappedInputs);
      case "COMPARE_BY_GENDER":
        return COMPARE_BY_GENDER(wrappedInputs);
      case "FILTER":
        return FILTER(wrappedInputs);
      case "FILTER_BY_SIZE":
        return FILTER_BY_SIZE(wrappedInputs);
      case "FILTER_BY_COLOR":
        return FILTER_BY_COLOR(wrappedInputs);
      case "FILTER_BY_TYPE":
        return FILTER_BY_TYPE(wrappedInputs);
      case "FILTER_BY_TASTE":
        return FILTER_BY_TASTE(wrappedInputs);
      case "FILTER_BY_AGE_GROUP":
        return FILTER_BY_AGE_GROUP(wrappedInputs);
      case "FILTER_BY_GENDER":
        return FILTER_BY_GENDER(wrappedInputs);
      case "UNION":
        return UNION(wrappedInputs);
      case "INTERSECTION":
        return INTERSECTION(wrappedInputs);
      case "DIFFERENCE":
        return DIFFERENCE(wrappedInputs);
      case "COMPLEMENT":
        return COMPLEMENT(wrappedInputs);
      case "SORT":
        return SORT(wrappedInputs);
      case "ALPHABETICAL_SORT":
        return ALPHABETICAL_SORT(wrappedInputs);
      default:
        throw new Error(`Unknown operation: ${operation}`);
    }
  }

  private invalidateDependentCache(nodeId: string): void {
    const dependents = this.nodeDependencies.get(nodeId) || new Set();
    const toInvalidate: string[] = [];

    for (const depId of dependents) {
      if (this.cache.has(depId)) {
        toInvalidate.push(depId);
      }
    }

    for (const id of toInvalidate) {
      this.cache.delete(id);
    }
  }

  private notifySubscribers(changedNodes: string[]): void {
    for (const nodeId of changedNodes) {
      const subscribers = this.subscriptions.get(nodeId);
      if (subscribers) {
        subscribers.forEach(callback => {
          const state = this.getNodeState(nodeId);
          callback(state);
        });
      }
    }
  }

  private getCachedState(nodeId: string, time: number): NodeState | null {
    const timeMap = this.cache.get(nodeId);
    if (!timeMap) return null;

    for (const cachedTime of Array.from(timeMap.keys()).sort()) {
      if (cachedTime > time) continue;
      return { status: "completed", value: timeMap.get(cachedTime) };
    }

    return null;
  }

  private buildDependencyGraph(): void {
    this.nodeDependencies.clear();

    for (const edge of this.edges.values()) {
      if (!this.nodeDependencies.has(edge.to)) {
        this.nodeDependencies.set(edge.to, new Set());
      }
      const existingDeps = this.nodeDependencies.get(edge.to) || new Set();
      existingDeps.add(edge.from);
      this.nodeDependencies.set(edge.to, existingDeps);
    }
  }

  private createMissingInputs(node: { operation: string }, inputCount: number): MissingInput[] {
    const missingInputs: MissingInput[] = [];
    for (let i = 0; i < inputCount; i++) {
      missingInputs.push({
        port: i,
        nodeId: "",
        description: this.getInputDescription(node.operation, i),
        childMessage: this.getChildMessageForMissingInput(node.operation, i)
      });
    }
    return missingInputs;
  }

  private getInputDescription(operation: string, inputIndex: number): string {
    const childMessages: Record<string, any> = {
      ADD: {
        0: "número",
        1: "número"
      },
      MULTIPLY: "número",
      SUBTRACT: "número",
      DIVIDE: "número",
      COMPARE: "número",
      FILTER_BY_COLOR: "color",
      FILTER_BY_SIZE: "tamaño",
      FILTER_BY_TYPE: "tipo",
      FILTER_BY_TASTE: "sabor",
      FILTER_BY_AGE_GROUP: "edad",
      FILTER_BY_GENDER: "género",
      UNION: "elementos",
      INTERSECTION: "elementos",
      DIFFERENCE: "elementos",
      COMPLEMENT: "elementos",
      SORT: "valores",
      ALPHABETICAL_SORT: "texto",
      FBY: "valor inicial y luego flujo",
      ACCUMULATE: "valor acumulado",
      NEXT: "siguiente valor",
      FIRST: "primer valor"
    };

    return childMessages[operation]?.[inputIndex] || "valor";
  }

  private getChildMessageForMissingInput(operation: string, inputIndex: number): string {
    const operationMessages: Record<string, string> = {
      ADD: "número",
      MULTIPLY: "número",
      SUBTRACT: "número",
      DIVIDE: "número",
      COMPARE: "número"
    };

    const typeMessages: Record<string, string> = {
      FILTER_BY_COLOR: "color",
      FILTER_BY_SIZE: "tamaño",
      FILTER_BY_TYPE: "tipo",
      FILTER_BY_TASTE: "sabor",
      FILTER_BY_AGE_GROUP: "edad",
      FILTER_BY_GENDER: "género"
    };

    if (operationMessages[operation]) {
      return operationMessages[operation];
    }

    if (operation.startsWith("FILTER_BY_")) {
      const property = operation.split("_BY_")[1].toLowerCase();
      return typeMessages[property] || "valor";
    }

    return "valor";
  }
}
