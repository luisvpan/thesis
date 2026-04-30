import type { DependencyGraph } from "./graph";
import type {
  RuntimeValue,
  CPAObject,
  ShapeValue,
  FoodValue,
  AbstractValue,
  ExecutionNode,
} from "./types";
import type { Statement, Expression, Literal, ObjectLiteral, ShapeTypeValue, FoodTypeValue } from "../analyzer/ast";
import { executeOperation } from "../operations";
import { toFraction } from "./rational";
import { RuntimeError } from "./errors";

export interface EvaluationResult {
  results: Map<string, RuntimeValue>;
  errors: RuntimeError[];
}

export class LazyEvaluator {
  private graph: DependencyGraph;
  private resultsCache: Map<string, RuntimeValue>;
  private evaluationStack: Set<string>;
  private pendingEvaluations: Map<string, Promise<RuntimeValue>>;

  constructor(graph: DependencyGraph, resultsCache?: Map<string, RuntimeValue>) {
    this.graph = graph;
    this.resultsCache = resultsCache ?? new Map();
    this.evaluationStack = new Set();
    this.pendingEvaluations = new Map();
  }

  /**
   * Evaluates all sink nodes, pulling dependencies lazily.
   * Uses memoization to ensure each node is evaluated exactly once.
   */
  async evaluate(): Promise<EvaluationResult> {
    const results = new Map<string, RuntimeValue>();
    const errors: RuntimeError[] = [];

    // Evaluate each sink node
    for (const sinkId of this.graph.sinkIds) {
      try {
        const result = await this.evaluateNode(sinkId);
        results.set(sinkId, result);
      } catch (err) {
        if (err instanceof RuntimeError) {
          errors.push(err);
        } else {
          throw err;
        }
      }
    }

    return { results, errors };
  }

  /**
   * Recursively evaluates a node, pulling dependencies first.
   * Memoization ensures each node is evaluated exactly once.
   */
  private async evaluateNode(nodeId: string): Promise<RuntimeValue> {
    // Check cache first (memoization)
    const cached = this.resultsCache.get(nodeId);
    if (cached !== undefined) {
      return cached;
    }

    // If already evaluating, wait for the result (handles duplicate deps)
    const pending = this.pendingEvaluations.get(nodeId);
    if (pending !== undefined) {
      return pending;
    }

    // Cycle detection during evaluation
    if (this.evaluationStack.has(nodeId)) {
      throw new RuntimeError(
        "EVALUATION_CYCLE",
        `Cycle detected during evaluation`,
        nodeId
      );
    }

    const node = this.graph.nodes.get(nodeId);
    if (!node) {
      throw new RuntimeError("UNDEFINED_REFERENCE", `Node not found: ${nodeId}`);
    }

    this.evaluationStack.add(nodeId);
    node.state = "evaluating";

    // Create and store the promise for this evaluation
    const evaluationPromise = this.doEvaluateNode(node);
    this.pendingEvaluations.set(nodeId, evaluationPromise);

    try {
      const result = await evaluationPromise;
      return result;
    } finally {
      this.evaluationStack.delete(nodeId);
      this.pendingEvaluations.delete(nodeId);
    }
  }

  private async doEvaluateNode(node: ExecutionNode): Promise<RuntimeValue> {
    // Get unique dependencies
    const uniqueDeps = [...new Set(node.dependencies)];

    // Evaluate all unique dependencies in parallel (pull model)
    const depEntries = await Promise.all(
      uniqueDeps.map(async (depId) => {
        const value = await this.evaluateNode(depId);
        return [depId, value] as const;
      })
    );
    const depValues = new Map(depEntries);

    // Evaluate this node
    const result = this.evaluateStatement(node.statement, depValues);

    // Cache result
    this.resultsCache.set(node.id, result);
    node.state = "completed";
    node.result = result;

    return result;
  }

  private evaluateStatement(
    stmt: Statement,
    deps: Map<string, RuntimeValue>
  ): RuntimeValue {
    switch (stmt.type) {
      case "SourceStatement":
        return this.evaluateLiteral(stmt.value);

      case "TransformStatement": {
        const args = stmt.arguments.map((arg) =>
          this.resolveExpression(arg, deps)
        );
        return executeOperation(stmt.operation, args);
      }

      case "SinkStatement":
        return deps.get(stmt.sourceIdentifier)!;
    }
  }

  private resolveExpression(
    expr: Expression,
    deps: Map<string, RuntimeValue>
  ): RuntimeValue {
    if (expr.type === "Identifier") {
      const value = deps.get(expr.name);
      if (value === undefined) {
        throw new RuntimeError(
          "UNDEFINED_REFERENCE",
          `Undefined reference: ${expr.name}`
        );
      }
      return value;
    }
    return this.evaluateLiteral(expr);
  }

  private evaluateLiteral(literal: Literal): RuntimeValue {
    switch (literal.type) {
      case "NumberLiteral":
        return {
          kind: "racional",
          value: toFraction(literal.value),
        };

      case "ObjectLiteral":
        return this.evaluateObjectLiteral(literal);

      case "ArrayLiteral":
        return {
          kind: "arreglo",
          elements: literal.elements.map((el) => {
            if (el.type === "Identifier") {
              // Identifiers in array literals are treated as "otro" values
              return { kind: "otro" as const, value: el.name };
            }
            return this.evaluateLiteral(el as Literal);
          }),
        };

      case "OtherLiteral":
        return {
          kind: "otro",
          value: literal.value,
        };
    }
  }

  private evaluateObjectLiteral(obj: ObjectLiteral): CPAObject {
    // Handle explicit category
    if (obj.category === "abstracto") {
      return {
        kind: "abstracto",
        category: "abstracto",
        objectType: "racional",
        value: toFraction(obj.value!),
      } as AbstractValue;
    }

    if (obj.category === "pictorico" || obj.objectType === "forma") {
      return {
        kind: "forma",
        category: "pictorico",
        subtype: obj.subtype as ShapeTypeValue,
        size: obj.size!,
        amount: toFraction(obj.amount ?? "1"),
      } as ShapeValue;
    }

    if (obj.category === "concreto" || obj.objectType === "comida") {
      return {
        kind: "comida",
        category: "concreto",
        subtype: obj.subtype as FoodTypeValue,
        color: obj.color!,
        amount: toFraction(obj.amount ?? "1"),
      } as FoodValue;
    }

    // Infer from objectType if it's a shape subtype
    if (
      obj.objectType === "circulo" ||
      obj.objectType === "cuadrado" ||
      obj.objectType === "triangulo" ||
      obj.objectType === "rectangulo" ||
      obj.objectType === "rombo" ||
      obj.objectType === "estrella" ||
      obj.objectType === "trapecio"
    ) {
      return {
        kind: "forma",
        category: "pictorico",
        subtype: obj.objectType,
        size: obj.size ?? "mediano",
        amount: toFraction(obj.amount ?? "1"),
      } as ShapeValue;
    }

    // Infer from objectType if it's a food subtype
    if (
      obj.objectType === "manzana" ||
      obj.objectType === "hamburguesa" ||
      obj.objectType === "uva" ||
      obj.objectType === "pasta" ||
      obj.objectType === "pera"
    ) {
      return {
        kind: "comida",
        category: "concreto",
        subtype: obj.objectType,
        color: obj.color ?? "verde",
        amount: toFraction(obj.amount ?? "1"),
      } as FoodValue;
    }

    // Default to abstract with value 0
    return {
      kind: "abstracto",
      category: "abstracto",
      objectType: "racional",
      value: toFraction(obj.value ?? "0"),
    } as AbstractValue;
  }
}
