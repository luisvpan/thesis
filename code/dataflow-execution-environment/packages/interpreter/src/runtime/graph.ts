import type { Program, Statement, Expression, IdentifierExpression } from "../analyzer/ast";
import type { ExecutionNode } from "./types";
import { RuntimeError } from "./errors";

export interface DependencyGraph {
  nodes: Map<string, ExecutionNode>;
  sinkIds: string[];
}

/**
 * Builds a dependency graph from the AST.
 * Identifies sink nodes as evaluation entry points.
 */
export function buildGraph(program: Program): DependencyGraph {
  const nodes = new Map<string, ExecutionNode>();
  const sinkIds: string[] = [];

  for (const stmt of program.statements) {
    const id = getStatementId(stmt);

    // Check for duplicate identifiers
    if (nodes.has(id)) {
      throw new RuntimeError("INVALID_ARGUMENT", `Duplicate identifier: ${id}`, id);
    }

    const deps = extractDependencies(stmt);

    nodes.set(id, {
      id,
      statement: stmt,
      dependencies: deps,
      dependents: [], // Will be populated below
      state: "pending",
    });

    if (stmt.type === "SinkStatement") {
      sinkIds.push(id);
    }
  }

  // Validate: check all dependencies exist
  for (const [id, node] of nodes) {
    for (const dep of node.dependencies) {
      if (!nodes.has(dep)) {
        throw new RuntimeError("UNDEFINED_REFERENCE", `Undefined reference: ${dep}`, id);
      }
    }
  }

  // Build reverse dependencies (dependents)
  for (const [id, node] of nodes) {
    for (const dep of node.dependencies) {
      nodes.get(dep)?.dependents.push(id);
    }
  }

  // Validate: check for circular dependencies
  validateNoCycles(nodes);

  return { nodes, sinkIds };
}

function getStatementId(stmt: Statement): string {
  return stmt.identifier;
}

function extractDependencies(stmt: Statement): string[] {
  switch (stmt.type) {
    case "SourceStatement":
      // Sources have no dependencies (only contain literals)
      return [];

    case "TransformStatement":
      // Extract identifier references from arguments
      return stmt.arguments
        .filter((arg): arg is IdentifierExpression => arg.type === "Identifier")
        .map((arg) => arg.name);

    case "SinkStatement":
      return [stmt.sourceIdentifier];
  }
}

function validateNoCycles(nodes: Map<string, ExecutionNode>): void {
  const visited = new Set<string>();
  const recursionStack = new Set<string>();

  function dfs(nodeId: string, path: string[]): void {
    if (recursionStack.has(nodeId)) {
      const cycleStart = path.indexOf(nodeId);
      const cycle = path.slice(cycleStart).concat(nodeId);
      throw new RuntimeError(
        "CIRCULAR_DEPENDENCY",
        `Circular dependency detected: ${cycle.join(" -> ")}`,
        nodeId
      );
    }

    if (visited.has(nodeId)) return;

    visited.add(nodeId);
    recursionStack.add(nodeId);

    const node = nodes.get(nodeId);
    if (node) {
      for (const dep of node.dependencies) {
        dfs(dep, [...path, nodeId]);
      }
    }

    recursionStack.delete(nodeId);
  }

  for (const nodeId of nodes.keys()) {
    if (!visited.has(nodeId)) {
      dfs(nodeId, []);
    }
  }
}
