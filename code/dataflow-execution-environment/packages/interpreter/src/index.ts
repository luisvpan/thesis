import { DataflowLexer } from "./analyzer/lexer";
import { parserInstance } from "./analyzer/parser";
import { visitorInstance } from "./analyzer/visitor";
import { buildGraph } from "./runtime/graph";
import { LazyEvaluator } from "./runtime/evaluator";
import { RuntimeError } from "./runtime/errors";
import type { Program } from "./analyzer/ast";
import type { RuntimeValue } from "./runtime/types";

// Re-export types
export * from "./analyzer/ast";
export * from "./runtime/types";
export * from "./runtime/errors";
export { DataflowLexer, allTokens } from "./analyzer/lexer";
export { DataflowParser, parserInstance } from "./analyzer/parser";
export { DataflowAstVisitor, visitorInstance } from "./analyzer/visitor";
export { buildGraph, type DependencyGraph } from "./runtime/graph";
export { LazyEvaluator, type EvaluationResult } from "./runtime/evaluator";
export { Interpreter, type EvaluationStats } from "./interpreter";
export { diffGraphs, type GraphDiff } from "./runtime/differ";
export { executeOperation } from "./operations";

export interface ParseResult {
  ast: Program | null;
  errors: ParseError[];
}

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
 * Parse a dataflow program string and return an AST.
 */
export function parse(input: string): ParseResult {
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
  const ast = visitorInstance.visit(cst) as Program;

  return { ast, errors: [] };
}

/**
 * Execute a dataflow program and return sink results.
 * Evaluation is lazy: starts from sinks and pulls dependencies.
 * Each node is evaluated exactly once (memoized).
 */
export async function execute(input: string): Promise<ExecuteResult> {
  const parseResult = parse(input);

  if (!parseResult.ast || parseResult.errors.length > 0) {
    return {
      results: new Map(),
      errors: parseResult.errors,
    };
  }

  try {
    const graph = buildGraph(parseResult.ast);
    const evaluator = new LazyEvaluator(graph);
    const evalResult = await evaluator.evaluate();

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
