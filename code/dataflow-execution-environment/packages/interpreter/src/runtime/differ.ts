import type { DependencyGraph } from "./graph";
import type { Statement } from "../analyzer/ast";

export interface GraphDiff {
  added: string[]; // New node IDs
  removed: string[]; // Deleted node IDs
  changed: string[]; // Modified node IDs (same ID, different content)
  unchanged: string[]; // Identical nodes
}

/**
 * Compares two dependency graphs and returns the diff.
 * Used for incremental re-evaluation to determine which nodes need re-evaluation.
 */
export function diffGraphs(
  oldGraph: DependencyGraph,
  newGraph: DependencyGraph
): GraphDiff {
  const diff: GraphDiff = { added: [], removed: [], changed: [], unchanged: [] };

  // Check nodes in the new graph
  for (const [id, newNode] of newGraph.nodes) {
    const oldNode = oldGraph.nodes.get(id);
    if (!oldNode) {
      diff.added.push(id);
    } else if (!statementsEqual(oldNode.statement, newNode.statement)) {
      diff.changed.push(id);
    } else {
      diff.unchanged.push(id);
    }
  }

  // Check for removed nodes
  for (const id of oldGraph.nodes.keys()) {
    if (!newGraph.nodes.has(id)) {
      diff.removed.push(id);
    }
  }

  return diff;
}

/**
 * Compares two statements for equality using deep comparison.
 */
function statementsEqual(a: Statement, b: Statement): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
