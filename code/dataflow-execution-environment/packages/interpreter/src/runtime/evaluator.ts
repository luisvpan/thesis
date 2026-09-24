// Evaluación dirigida por demanda — LANGUAGE_SPEC.md §2.3
//
// Parte de los `sink` y tira hacia atrás de las dependencias. Cada nodo se
// evalúa una sola vez (§2.3.2) y cada salida se evalúa aislada: un error en una
// no impide obtener el valor de las demás (§2.3.1, §4.3).

import type {
  CriterionLiteral,
  DataLiteral,
  Literal,
  Statement,
} from "../analyzer/ast";
import { executeOperation } from "../operations";
import { NULO, bag } from "./bag";
import { DataflowError } from "./errors";
import type { DependencyGraph } from "./graph";
import { toFraction } from "./rational";
import type {
  CPACategory,
  Criterion,
  CriterionValue,
  Entry,
  ExecutionNode,
  RuntimeValue,
} from "./types";

export interface EvaluationResult {
  results: Map<string, RuntimeValue>;
  errors: DataflowError[];
}

export class LazyEvaluator {
  private graph: DependencyGraph;
  private resultsCache: Map<string, RuntimeValue>;
  private pendingEvaluations: Map<string, Promise<RuntimeValue>>;
  /** Las salidas a calcular; por defecto, todas las del programa. */
  private sinkIds: string[];
  /** La salida cuyo cálculo está en curso, para situar los errores (§4). */
  private currentSinkId?: string;

  constructor(
    graph: DependencyGraph,
    resultsCache?: Map<string, RuntimeValue>,
    sinkIds?: string[]
  ) {
    this.graph = graph;
    this.resultsCache = resultsCache ?? new Map();
    this.pendingEvaluations = new Map();
    this.sinkIds = sinkIds ?? graph.sinkIds;
  }

  /** EvaluarPrograma(programa) → (valores, errores) — §2.3.1 */
  async evaluate(): Promise<EvaluationResult> {
    const results = new Map<string, RuntimeValue>();
    const errors: DataflowError[] = [];

    for (const sinkId of this.sinkIds) {
      this.currentSinkId = sinkId;
      try {
        results.set(sinkId, await this.evaluateNode(sinkId));
      } catch (err) {
        if (err instanceof DataflowError) {
          errors.push(err.situate({ sinkIds: [sinkId] }));
        } else {
          throw err;
        }
      }
    }
    this.currentSinkId = undefined;

    // Los transforms ya calculados se exponen para que la interfaz pueda
    // mostrar resultados intermedios sobre las aristas.
    for (const [nodeId, node] of this.graph.nodes) {
      if (node.statement.type !== "TransformStatement") continue;
      const cached = this.resultsCache.get(nodeId);
      if (cached !== undefined) results.set(nodeId, cached);
    }

    return { results, errors };
  }

  /** EvaluarNodo(id) → valor — §2.3.2 */
  private async evaluateNode(nodeId: string): Promise<RuntimeValue> {
    const cached = this.resultsCache.get(nodeId);
    if (cached !== undefined) return cached;

    // Una evaluación en curso se comparte, de modo que un nodo del que dependen
    // varios se calcula una sola vez.
    const pending = this.pendingEvaluations.get(nodeId);
    if (pending !== undefined) return pending;

    const node = this.graph.nodes.get(nodeId);
    if (!node) {
      throw new DataflowError("UNDEFINED_REFERENCE", `No existe el nodo '${nodeId}'`, {
        nodeId,
      });
    }

    node.state = "evaluating";
    const evaluation = this.doEvaluateNode(node);
    this.pendingEvaluations.set(nodeId, evaluation);

    try {
      return await evaluation;
    } finally {
      this.pendingEvaluations.delete(nodeId);
    }
  }

  private async doEvaluateNode(node: ExecutionNode): Promise<RuntimeValue> {
    // Las entradas se resuelven antes que el nodo; las independientes, en paralelo.
    const uniqueDeps = [...new Set(node.dependencies)];
    const depEntries = await Promise.all(
      uniqueDeps.map(async (depId) => [depId, await this.evaluateNode(depId)] as const)
    );

    const result = this.evaluateStatement(node.statement, new Map(depEntries));

    this.resultsCache.set(node.id, result);
    node.state = "completed";
    node.result = result;

    return result;
  }

  /** EvaluarSentencia(nodo, entradas) → valor — §2.3.3 */
  private evaluateStatement(stmt: Statement, deps: Map<string, RuntimeValue>): RuntimeValue {
    switch (stmt.type) {
      // Un `source` aporta su valor directamente: no lo calcula a partir de
      // otros nodos.
      case "SourceStatement":
        return stmt.value ? evaluateLiteral(stmt.value) : NULO;

      case "TransformStatement": {
        if (!stmt.operation) return NULO;

        const args = stmt.arguments.map((argument) => deps.get(argument.name) ?? NULO);

        try {
          return executeOperation(stmt.operation, args);
        } catch (err) {
          if (err instanceof DataflowError) {
            const cause =
              err.argumentIndex !== undefined
                ? stmt.arguments[err.argumentIndex]?.name
                : undefined;
            throw err.situate({
              nodeId: stmt.identifier,
              causeNodeId: cause ?? stmt.identifier,
              sinkIds: this.currentSinkId ? [this.currentSinkId] : [],
            });
          }
          throw err;
        }
      }

      case "SinkStatement":
        if (!stmt.sourceIdentifier) return NULO;
        return deps.get(stmt.sourceIdentifier) ?? NULO;
    }
  }
}

// =============================================================================
// Literales → valores
// =============================================================================

export function evaluateLiteral(literal: Literal): RuntimeValue {
  switch (literal.type) {
    case "DataLiteral":
      return bag([toEntry(literal)]);

    case "GroupLiteral":
      return bag(literal.elements.map(toEntry));

    case "CriterionLiteral":
      return toCriterion(literal);
  }
}

/**
 * Una entrada de la bolsa. La identidad ya viene validada por la pasada
 * estática (§4.2.8: ningún componente en blanco) y por la gramática (la
 * categoría es una de las tres).
 */
function toEntry(literal: DataLiteral): Entry {
  const attributes: Record<string, string> = {};
  for (const property of literal.attributes) {
    if (typeof property.value === "string") attributes[property.key] = property.value;
  }

  return {
    category: literal.category as CPACategory,
    type: literal.objType,
    subtype: literal.subtype,
    attributes,
    quantity: toFraction(literal.quantity || "1"),
  };
}

function toCriterion(literal: CriterionLiteral): Criterion {
  const values: Record<string, CriterionValue> = {};
  for (const property of literal.values) {
    values[property.key] = property.value;
  }

  return {
    kind: "criterio",
    subtype: literal.sourceType,
    properties: [...literal.properties],
    values,
  };
}
