import { analyze } from "./analyzer/static-analysis";
import type { Program } from "./program";
import { deserialize, parseToAst } from "./serializer";
import { diffGraphs } from "./runtime/differ";
import type { DataflowError } from "./runtime/errors";
import { LazyEvaluator } from "./runtime/evaluator";
import { buildGraph, type DependencyGraph } from "./runtime/graph";
import type { RuntimeValue } from "./runtime/types";

export interface ExecuteResult {
  results: Map<string, RuntimeValue>;
  /** Una sola forma para las tres fases: sintaxis, estática y ejecución (§4). */
  errors: DataflowError[];
}

export interface EvaluationStats {
  /** Nodos calculados en esta ejecución. */
  evaluated: number;
  /** Nodos reutilizados de la caché. */
  cached: number;
  /** Nodos del grafo. */
  total: number;
}

/**
 * Mantiene estado entre ejecuciones para reevaluar de forma incremental: al
 * cambiar el programa, solo se invalidan los nodos afectados y sus dependientes
 * (§2.4: el valor de un nodo depende solo de su sentencia y de sus entradas).
 */
export class Interpreter {
  private currentGraph: DependencyGraph | null = null;
  private resultsCache: Map<string, RuntimeValue> = new Map();
  private lastStats: EvaluationStats = { evaluated: 0, cached: 0, total: 0 };

  /**
   * Ejecuta un programa, sea texto o un `Program` ya construido.
   *
   * El orden es el de §4: sintaxis, luego estática, luego evaluación. Un error
   * estático invalida el programa completo, así que no se evalúa ningún nodo.
   */
  async execute(input: string | Program): Promise<ExecuteResult> {
    const sourceCode = typeof input === "string" ? input : deserialize(input);

    const { ast, errors: syntaxErrors } = parseToAst(sourceCode);
    if (!ast || syntaxErrors.length > 0) {
      return { results: new Map(), errors: syntaxErrors };
    }

    const newGraph = buildGraph(ast);

    // Un error estático no apaga el programa entero: apaga las salidas en cuyo
    // camino está el nodo culpable (§4.2). Las demás se calculan igual.
    const staticErrors = analyze(newGraph);
    const blocked = new Set(staticErrors.flatMap((error) => error.sinkIds));

    if (this.currentGraph) {
      const diff = diffGraphs(this.currentGraph, newGraph);
      this.invalidateNodes(diff.changed, diff.removed, newGraph);
    }

    const cachedCount = this.countCachedNodes(newGraph);
    const totalNodes = newGraph.nodes.size;

    this.currentGraph = newGraph;

    const evaluator = new LazyEvaluator(
      newGraph,
      this.resultsCache,
      newGraph.sinkIds.filter((sinkId) => !blocked.has(sinkId))
    );
    const { results, errors } = await evaluator.evaluate();

    this.lastStats = {
      evaluated: totalNodes - cachedCount,
      cached: cachedCount,
      total: totalNodes,
    };

    return { results, errors: [...staticErrors, ...errors] };
  }

  /** Invalida los nodos cambiados y, en cascada, todo lo que depende de ellos. */
  private invalidateNodes(changed: string[], removed: string[], graph: DependencyGraph): void {
    const toInvalidate = new Set<string>();
    const queue = [...changed, ...removed];

    while (queue.length > 0) {
      const id = queue.shift()!;
      if (toInvalidate.has(id)) continue;
      toInvalidate.add(id);
      queue.push(...(graph.nodes.get(id)?.dependents ?? []));
    }

    for (const id of toInvalidate) {
      this.resultsCache.delete(id);
    }
  }

  reset(): void {
    this.currentGraph = null;
    this.resultsCache.clear();
    this.lastStats = { evaluated: 0, cached: 0, total: 0 };
  }

  getCacheSize(): number {
    return this.resultsCache.size;
  }

  getEvaluationStats(): EvaluationStats {
    return { ...this.lastStats };
  }

  private countCachedNodes(graph: DependencyGraph): number {
    let count = 0;
    for (const nodeId of graph.nodes.keys()) {
      if (this.resultsCache.has(nodeId)) count++;
    }
    return count;
  }
}
