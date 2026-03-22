import { DataflowGraph } from "../graph/dataflow-graph.js";
import {
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  COMPARE
} from "../operations/numeric.js";
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
import { LRUCache } from "../utils/lru-cache.js";

export class DemandDrivenEvaluator {
  private cache = new LRUCache<string, Map<number, unknown>>(1000);
  private cacheHits = 0;
  private cacheMisses = 0;
  private executionOrder: string[] = [];
  private nodeEvaluations = new Map<string, { value: unknown; timestep: number }>();

  async evaluate(nodeId: string, time: number, graph: DataflowGraph): Promise<unknown> {
    const nodeCache = this.cache.get(nodeId);
    if (nodeCache && nodeCache.has(time)) {
      this.cacheHits++;
      return nodeCache.get(time);
    }
    this.cacheMisses++;

    const node = graph.getNode(nodeId);
    if (!node) throw new Error(`Node ${nodeId} not found`);

    let value: unknown;
    if (node.type === "DataSource") {
      value = this.wrapDataSourceValue(node, time);
    } else if (node.type === "Transformation") {
      const inputNodes = graph.getInputs(nodeId);
      value = await this.evaluateOperation(node.operation, inputNodes, time, graph, nodeId);
    } else if (node.type === "Output") {
      const inputNodes = graph.getInputs(nodeId);
      if (inputNodes.length === 0) {
        throw new Error(`Output node ${nodeId} has no input`);
      }
      value = await this.evaluate(inputNodes[0].id, time, graph);
    }

    const existingCache = this.cache.get(nodeId);
    if (existingCache) {
      existingCache.set(time, value);
    } else {
      const nodeCache = new Map<number, unknown>();
      nodeCache.set(time, value);
      this.cache.set(nodeId, nodeCache);
    }

    this.executionOrder.push(nodeId);
    this.nodeEvaluations.set(nodeId, { value, timestep: time });

    return value;
  }

  private wrapDataSourceValue(node: { type: "DataSource"; dataType: string | { kind: string; elementType?: any }; value: unknown }, time: number): unknown {
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
          if (typeof numValue === 'object' && numValue !== null && 'kind' in numValue && 'elements' in numValue) {
            return numValue;
          }
          
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
            return elem;
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

  private async evaluateOperation(operation: string, inputs: Array<{ id: string; value: unknown }>, time: number, graph: DataflowGraph, currentNodeId: string): Promise<unknown> {
    switch (operation) {
      case "NEXT":
        return await NEXT(inputs, time, graph, this, currentNodeId);
      case "FIRST":
        return await FIRST(inputs, time, graph, this, currentNodeId);
      case "FBY":
        return await FBY(inputs, time, graph, this, currentNodeId);
      case "ACCUMULATE":
        return await ACCUMULATE(inputs, time, graph, this, currentNodeId);
      case "ADD":
        return ADD(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "SUBTRACT":
        return SUBTRACT(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "MULTIPLY":
        return MULTIPLY(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "DIVIDE":
        return DIVIDE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPARE":
        return COMPARE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));

      case "AND":
        return AND(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "OR":
        return OR(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "NOT":
        return NOT(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));

      case "COMPARE_BY_SIZE":
        return COMPARE_BY_SIZE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPARE_BY_COLOR":
        return COMPARE_BY_COLOR(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPARE_BY_TYPE":
        return COMPARE_BY_TYPE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPARE_BY_TASTE":
        return COMPARE_BY_TASTE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPARE_BY_AGE_GROUP":
        return COMPARE_BY_AGE_GROUP(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPARE_BY_GENDER":
        return COMPARE_BY_GENDER(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));

      case "FILTER":
        return FILTER(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "FILTER_BY_SIZE":
        return FILTER_BY_SIZE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "FILTER_BY_COLOR":
        return FILTER_BY_COLOR(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "FILTER_BY_TYPE":
        return FILTER_BY_TYPE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "FILTER_BY_TASTE":
        return FILTER_BY_TASTE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "FILTER_BY_AGE_GROUP":
        return FILTER_BY_AGE_GROUP(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "FILTER_BY_GENDER":
        return FILTER_BY_GENDER(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));

      case "UNION":
        return UNION(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "INTERSECTION":
        return INTERSECTION(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "DIFFERENCE":
        return DIFFERENCE(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "COMPLEMENT":
        return COMPLEMENT(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "SORT":
        return SORT(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));
      case "ALPHABETICAL_SORT":
        return ALPHABETICAL_SORT(await Promise.all(inputs.map(async input => ({
          id: input.id,
          value: await this.evaluate(input.id, time, graph)
        }))));

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
    this.executionOrder = [];
    this.nodeEvaluations.clear();
  }

  getExecutionTrace(): { executionOrder: string[]; nodeEvaluations: Record<string, { value: unknown; timestep: number }> } {
    return {
      executionOrder: [...this.executionOrder],
      nodeEvaluations: Object.fromEntries(this.nodeEvaluations)
    };
  }
}
