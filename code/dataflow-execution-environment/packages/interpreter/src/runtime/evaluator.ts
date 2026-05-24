import type { DependencyGraph } from "./graph";
import type {
  RuntimeValue,
  CPAObject,
  CPACategory,
  ExecutionNode,
} from "./types";
import type { Statement, Expression, Literal, ObjectLiteral } from "../analyzer/ast";
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
        // Handle incomplete source statements
        if (!stmt.value) {
          return { kind: "otro", value: "" };
        }
        return this.evaluateSourceStatementValue(stmt.value, deps);

      case "TransformStatement": {
        // Handle incomplete transform statements
        if (!stmt.operation) {
          return { kind: "otro", value: "" };
        }
        const args = stmt.arguments.map((arg) =>
          this.resolveExpression(arg, deps)
        );
        return executeOperation(stmt.operation, args);
      }

      case "SinkStatement":
        // Handle incomplete sink statements
        if (!stmt.sourceIdentifier) {
          return { kind: "otro", value: "" };
        }
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

  /**
   * Valor de un `source`: literales de arreglo con identificadores resuelven contra `deps`
   * (nodos fuente referenciados, declarados en el grafo antes que este `source`).
   */
  private evaluateSourceStatementValue(
    literal: Literal,
    deps: Map<string, RuntimeValue>
  ): RuntimeValue {
    if (literal.type === "ArrayLiteral") {
      return {
        kind: "arreglo",
        elements: literal.elements.map((el) => {
          if (el.type === "Identifier") {
            const v = deps.get(el.name);
            if (v === undefined) {
              throw new RuntimeError(
                "UNDEFINED_REFERENCE",
                `Undefined reference in array literal: ${el.name}`,
                undefined
              );
            }
            return v;
          }
          return this.evaluateLiteral(el as Literal);
        }),
      };
    }
    return this.evaluateLiteral(literal);
  }

  private evaluateLiteral(literal: Literal): RuntimeValue {
    switch (literal.type) {
      case "StringLiteral":
        return {
          kind: "otro",
          value: literal.value,
        };

      case "ObjectLiteral":
        return this.evaluateObjectLiteral(literal);

      case "ArrayLiteral":
        return {
          kind: "arreglo",
          elements: literal.elements.map((el) => {
            if (el.type === "Identifier") {
              throw new RuntimeError(
                "INVALID_ARGUMENT",
                "Array literal with identifiers must be bound in a source statement so references are resolved (e.g. source z = [a, b]); bare identifiers in inline arrays are not supported here.",
                undefined
              );
            }
            return this.evaluateLiteral(el as Literal);
          }),
        };
    }
  }

  /**
   * Evaluates an ObjectLiteral to a GenericCPAObject.
   * Uses the new unified format with properties array.
   */
  private evaluateObjectLiteral(obj: ObjectLiteral): CPAObject {
    // Extract properties from the ObjectLiteral
    const props = new Map<string, string>();
    for (const prop of obj.properties) {
      props.set(prop.key, prop.value);
    }

    // Get required fields
    const category = props.get("category") as CPACategory | undefined;
    const type = props.get("type");
    const subtype = props.get("subtype");
    const quantity = props.get("quantity");

    // Validate required fields (with defaults for backwards compatibility)
    if (!category) {
      throw new RuntimeError(
        "INVALID_OBJECT",
        "Object must have a 'category' property"
      );
    }

    if (!type) {
      throw new RuntimeError(
        "INVALID_OBJECT",
        "Object must have a 'type' property"
      );
    }

    if (!subtype) {
      throw new RuntimeError(
        "INVALID_OBJECT",
        "Object must have a 'subtype' property"
      );
    }

    // Extract additional attributes (everything except category, type, subtype, quantity)
    const attributes: Record<string, string> = {};
    for (const [key, value] of props) {
      if (!["category", "type", "subtype", "quantity"].includes(key)) {
        attributes[key] = value;
      }
    }

    return {
      kind: "cpa",
      category,
      type,
      subtype,
      quantity: toFraction(quantity ?? "1"),
      attributes,
    };
  }
}
