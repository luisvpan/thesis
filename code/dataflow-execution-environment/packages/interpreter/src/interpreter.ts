import { DataflowLexer } from "./analyzer/lexer";
import { parserInstance } from "./analyzer/parser";
import { visitorInstance } from "./analyzer/visitor";
import { buildGraph, type DependencyGraph } from "./runtime/graph";
import { LazyEvaluator } from "./runtime/evaluator";
import { diffGraphs } from "./runtime/differ";
import { RuntimeError } from "./runtime/errors";
import type { RuntimeValue } from "./runtime/types";
import type { Program as ASTProgram } from "./analyzer/ast";
import type { Program } from "./program";
import { deserialize } from "./serializer";

export interface ParseError {
  message: string;
  line?: number;
  column?: number;
}

export interface ExecuteResult {
  results: Map<string, RuntimeValue>;
  errors: Array<ParseError | RuntimeError>;
}

/**
 * Interpreter class that maintains state between executions.
 * Enables true incremental re-evaluation: when source code changes,
 * only affected nodes are re-evaluated, reusing cached results for unchanged nodes.
 */
export interface EvaluationStats {
  /** Number of nodes that were newly evaluated */
  evaluated: number;
  /** Number of nodes that were reused from cache */
  cached: number;
  /** Total nodes in the graph */
  total: number;
}

export class Interpreter {
  private currentGraph: DependencyGraph | null = null;
  private resultsCache: Map<string, RuntimeValue> = new Map();
  private lastStats: EvaluationStats = { evaluated: 0, cached: 0, total: 0 };

  /**
   * Execute a dataflow program. If this is not the first execution,
   * performs incremental re-evaluation by diffing against the previous graph
   * and only invalidating changed nodes and their dependents.
   *
   * @param input - Either a dataflow source string or a Program object
   */
  async execute(input: string | Program): Promise<ExecuteResult> {
    // Convert Program to string if needed
    const sourceCode = typeof input === "string" ? input : deserialize(input);

    const parseResult = this.parse(sourceCode);

    if (!parseResult.ast || parseResult.errors.length > 0) {
      return {
        results: new Map(),
        errors: parseResult.errors,
      };
    }

    try {
      const newGraph = buildGraph(parseResult.ast);

      if (this.currentGraph) {
        // Incremental: diff and invalidate
        const diff = diffGraphs(this.currentGraph, newGraph);
        this.invalidateNodes(diff.changed, diff.removed, newGraph);
      }

      // Track how many nodes are already cached (will be reused)
      const cachedCount = this.countCachedNodes(newGraph);
      const totalNodes = newGraph.nodes.size;

      this.currentGraph = newGraph;

      const evaluator = new LazyEvaluator(newGraph, this.resultsCache);
      const evalResult = await evaluator.evaluate();

      // Update stats
      this.lastStats = {
        evaluated: totalNodes - cachedCount,
        cached: cachedCount,
        total: totalNodes,
      };

      return {
        results: evalResult.results,
        errors: evalResult.errors,
      };
    } catch (err) {
      if (err instanceof RuntimeError) {
        return {
          results: new Map(),
          errors: [err],
        };
      }
      throw err;
    }
  }

  /**
   * Invalidate nodes and cascade invalidation to their dependents.
   * This removes cached results for nodes that need re-evaluation.
   */
  private invalidateNodes(
    changed: string[],
    removed: string[],
    graph: DependencyGraph
  ): void {
    const toInvalidate = new Set<string>();

    // Invalidate changed/removed nodes and all their dependents (cascade)
    const queue = [...changed, ...removed];
    while (queue.length > 0) {
      const id = queue.shift()!;
      if (toInvalidate.has(id)) continue;
      toInvalidate.add(id);

      // Add all nodes that depend on this one
      const node = graph.nodes.get(id);
      if (node) {
        queue.push(...node.dependents);
      }
    }

    // Remove invalidated entries from cache
    for (const id of toInvalidate) {
      this.resultsCache.delete(id);
    }
  }

  /**
   * Reset all interpreter state, clearing the graph and cache.
   */
  reset(): void {
    this.currentGraph = null;
    this.resultsCache.clear();
    this.lastStats = { evaluated: 0, cached: 0, total: 0 };
  }

  /**
   * Get the number of cached results (useful for testing/debugging).
   */
  getCacheSize(): number {
    return this.resultsCache.size;
  }

  /**
   * Get evaluation statistics from the last execution.
   * Shows how many nodes were evaluated vs reused from cache.
   */
  getEvaluationStats(): EvaluationStats {
    return { ...this.lastStats };
  }

  /**
   * Count how many nodes from the new graph are already in the cache.
   */
  private countCachedNodes(graph: DependencyGraph): number {
    let count = 0;
    for (const nodeId of graph.nodes.keys()) {
      if (this.resultsCache.has(nodeId)) {
        count++;
      }
    }
    return count;
  }

  /**
   * Parse a dataflow program string and return an AST.
   */
  private parse(input: string): { ast: ASTProgram | null; errors: ParseError[] } {
    const errors: ParseError[] = [];

    // Tokenize
    const lexResult = DataflowLexer.tokenize(input);

    if (lexResult.errors.length > 0) {
      for (const error of lexResult.errors) {
        errors.push({
          message: error.message,
          line: error.line,
          column: error.column,
        });
      }
      return { ast: null, errors };
    }

    // Parse
    parserInstance.input = lexResult.tokens;
    const cst = parserInstance.program();

    if (parserInstance.errors.length > 0) {
      for (const error of parserInstance.errors) {
        errors.push({
          message: error.message,
          line: error.token.startLine,
          column: error.token.startColumn,
        });
      }
      return { ast: null, errors };
    }

    // Build AST
    const ast = visitorInstance.visit(cst) as ASTProgram;

    return { ast, errors: [] };
  }
}
