import type { Statement } from "../ast/ast-types.js";

export function detectCycles(statements: Statement[]): boolean {
  const adj = new Map<string, string[]>();
  const inDegree = new Map<string, number>();
  const allIds = new Set<string>();

  for (const stmt of statements) {
    allIds.add(stmt.id);
    inDegree.set(stmt.id, 0);
  }

  for (const stmt of statements) {
    if (stmt.type === "TransformStatement") {
      for (const input of stmt.inputs) {
        if (!allIds.has(input)) continue;
        if (!adj.has(input)) {
          adj.set(input, []);
        }
        adj.get(input)!.push(stmt.id);
        inDegree.set(stmt.id, (inDegree.get(stmt.id) || 0) + 1);
      }
    } else if (stmt.type === "OutputStatement") {
      if (!allIds.has(stmt.input)) continue;
      if (!adj.has(stmt.input)) {
        adj.set(stmt.input, []);
      }
      adj.get(stmt.input)!.push(stmt.id);
      inDegree.set(stmt.id, (inDegree.get(stmt.id) || 0) + 1);
    }
  }

  const queue: string[] = [];
  for (const [id, degree] of inDegree.entries()) {
    if (degree === 0) {
      queue.push(id);
    }
  }

  let visited = 0;
  while (queue.length > 0) {
    const node = queue.shift()!;
    visited++;

    if (adj.has(node)) {
      for (const neighbor of adj.get(node)!) {
        inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
        if (inDegree.get(neighbor) === 0) {
          queue.push(neighbor);
        }
      }
    }
  }

  return visited !== allIds.size;
}
