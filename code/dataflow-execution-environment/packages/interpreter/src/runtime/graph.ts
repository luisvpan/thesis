// El programa como grafo — LANGUAGE_SPEC.md §2.1
//
// Construir el grafo no valida nada: la bien-formación (nombres únicos,
// referencias resueltas, aciclicidad) la comprueba la pasada estática (§2.2,
// §4.2), que trabaja sobre este grafo.

import type { Program, Statement } from "../analyzer/ast";
import type { ExecutionNode } from "./types";

export interface DependencyGraph {
  nodes: Map<string, ExecutionNode>;
  sinkIds: string[];
  /** Nombres declarados más de una vez, en orden de reaparición (§4.2.1). */
  duplicateIds: string[];
}

/**
 * Un nodo depende de los nodos que menciona por su nombre: un `transform`, de
 * sus argumentos; un `sink`, del nodo que expone. **Un `source` es entrada
 * pura**: declara literales y no referencia a nadie (§2.1).
 */
export function extractDependencies(stmt: Statement): string[] {
  switch (stmt.type) {
    case "SourceStatement":
      return [];

    case "TransformStatement":
      return stmt.arguments.map((argument) => argument.name);

    case "SinkStatement":
      return stmt.sourceIdentifier ? [stmt.sourceIdentifier] : [];
  }
}

export function buildGraph(program: Program): DependencyGraph {
  const nodes = new Map<string, ExecutionNode>();
  const sinkIds: string[] = [];
  const duplicateIds: string[] = [];

  for (const statement of program.statements) {
    const id = statement.identifier;

    if (nodes.has(id)) {
      // Gana la primera declaración; la repetición la reporta la pasada estática.
      duplicateIds.push(id);
      continue;
    }

    nodes.set(id, {
      id,
      statement,
      dependencies: extractDependencies(statement),
      dependents: [],
      state: "pending",
    });

    if (statement.type === "SinkStatement") {
      sinkIds.push(id);
    }
  }

  for (const [id, node] of nodes) {
    for (const dependency of node.dependencies) {
      nodes.get(dependency)?.dependents.push(id);
    }
  }

  return { nodes, sinkIds, duplicateIds };
}
