import { DataflowGraph } from "../graph/dataflow-graph.js";
import { ADD, SUBTRACT, MULTIPLY, DIVIDE, COMPARE } from "../operations/numeric.js";
import { AND, OR, NOT } from "../operations/boolean.js";
import {
  COMPARE_BY_SIZE,
  COMPARE_BY_COLOR,
  COMPARE_BY_TYPE,
  COMPARE_BY_TASTE,
  COMPARE_BY_AGE_GROUP,
  COMPARE_BY_GENDER
} from "../operations/comparison.js";
import {
  FILTER,
  FILTER_BY_SIZE,
  FILTER_BY_COLOR,
  FILTER_BY_TYPE,
  FILTER_BY_TASTE,
  FILTER_BY_AGE_GROUP,
  FILTER_BY_GENDER
} from "../operations/filtering.js";
import {
  UNION,
  INTERSECTION,
  DIFFERENCE,
  COMPLEMENT,
  SORT,
  ALPHABETICAL_SORT
} from "../operations/sets.js";
import { NEXT, FIRST, FBY, ACCUMULATE } from "../operations/temporal.js";
import type { Natural, Integer, Decimal, Fraction } from "@dataflow/shared/types";

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
      value = this.wrapDataSourceValue(node);
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

  private wrapDataSourceValue(node: { type: "DataSource"; dataType: string | { kind: string; elementType?: any }; value: unknown }): unknown {
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
      default:
        if (dataType.startsWith("stream")) {
          return node.value;
        }
        if (dataType.startsWith("set")) {
          return { kind: "set", elements: numValue as unknown[] };
        }
        return node.value;
    }
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

      case "AND":
        return AND(evaluatedInputs);
      case "OR":
        return OR(evaluatedInputs);
      case "NOT":
        return NOT(evaluatedInputs);

      case "COMPARE_BY_SIZE":
        return COMPARE_BY_SIZE(evaluatedInputs);
      case "COMPARE_BY_COLOR":
        return COMPARE_BY_COLOR(evaluatedInputs);
      case "COMPARE_BY_TYPE":
        return COMPARE_BY_TYPE(evaluatedInputs);
      case "COMPARE_BY_TASTE":
        return COMPARE_BY_TASTE(evaluatedInputs);
      case "COMPARE_BY_AGE_GROUP":
        return COMPARE_BY_AGE_GROUP(evaluatedInputs);
      case "COMPARE_BY_GENDER":
        return COMPARE_BY_GENDER(evaluatedInputs);

      case "FILTER":
        return FILTER(evaluatedInputs);
      case "FILTER_BY_SIZE":
        return FILTER_BY_SIZE(evaluatedInputs);
      case "FILTER_BY_COLOR":
        return FILTER_BY_COLOR(evaluatedInputs);
      case "FILTER_BY_TYPE":
        return FILTER_BY_TYPE(evaluatedInputs);
      case "FILTER_BY_TASTE":
        return FILTER_BY_TASTE(evaluatedInputs);
      case "FILTER_BY_AGE_GROUP":
        return FILTER_BY_AGE_GROUP(evaluatedInputs);
      case "FILTER_BY_GENDER":
        return FILTER_BY_GENDER(evaluatedInputs);

      case "UNION":
        return UNION(evaluatedInputs);
      case "INTERSECTION":
        return INTERSECTION(evaluatedInputs);
      case "DIFFERENCE":
        return DIFFERENCE(evaluatedInputs);
      case "COMPLEMENT":
        return COMPLEMENT(evaluatedInputs);
      case "SORT":
        return SORT(evaluatedInputs);
      case "ALPHABETICAL_SORT":
        return ALPHABETICAL_SORT(evaluatedInputs);

      case "NEXT":
        return NEXT(evaluatedInputs, time, graph);
      case "FIRST":
        return FIRST(evaluatedInputs, time, graph);
      case "FBY":
        return FBY(evaluatedInputs, time, graph);
      case "ACCUMULATE":
        return ACCUMULATE(evaluatedInputs, time, graph);

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
