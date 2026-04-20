/**
 * Serializa el programa visual (ReactFlow) al formato DataflowProgram del backend.
 */

import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/NodeContext";

// Tipos del backend (simplificados para evitar dependencia de @dataflow/shared)
type DataSourceNode = {
  id: string;
  type: "DataSource";
  dataType: "natural";
  value: { kind: "natural"; value: number };
};

type TransformationNode = {
  id: string;
  type: "Transformation";
  dataType: "natural";
  operation: string;
  inputs: string[];
};

type OutputNode = {
  id: string;
  type: "Output";
  dataType: "natural";
  input: string;
};

type ProgramNode = DataSourceNode | TransformationNode | OutputNode;

type DataflowEdge = {
  id: string;
  from: string;
  to: string;
  toPort?: number;
};

export type DataflowProgram = {
  metadata: {
    programId: string;
    activityId?: string;
    level?: number;
  };
  graph: {
    nodes: ProgramNode[];
    edges: DataflowEdge[];
  };
};

/** Mapeo de operadores frontend → backend */
const OPERATOR_MAP: Record<string, string> = {
  adicion: "ADD",
  sustraccion: "SUBTRACT",
  multiplicacion: "MULTIPLY",
  division: "DIVIDE",
};

/**
 * Convierte nodos y edges de ReactFlow al formato DataflowProgram del backend.
 */
export function serializeProgram(
  nodes: DataflowNode[],
  edges: Edge[]
): DataflowProgram {
  const programNodes: ProgramNode[] = [];
  const programEdges: DataflowEdge[] = [];

  if (nodes.length === 0) {
    return {
      metadata: { programId: `canvas-${Date.now()}` },
      graph: { nodes: [], edges: [] },
    };
  }

  // 1. Convertir number nodes → DataSourceNode
  for (const node of nodes) {
    if (node.type === "number") {
      const value = (node.data as { value?: number }).value ?? 0;
      programNodes.push({
        id: node.id,
        type: "DataSource",
        dataType: "natural",
        value: { kind: "natural", value },
      });
    }
  }

  // 2. Convertir operator nodes → TransformationNode
  for (const node of nodes) {
    if (node.type === "operator") {
      const operator = (node.data as { operator?: string }).operator ?? "adicion";

      // Encontrar inputs desde edges (a=puerto 0, b=puerto 1)
      const inputEdges = edges.filter((e) => e.target === node.id);
      const inputs: string[] = [];

      const edgeA = inputEdges.find((e) => e.targetHandle === "a");
      const edgeB = inputEdges.find((e) => e.targetHandle === "b");

      if (edgeA) inputs.push(edgeA.source);
      if (edgeB) inputs.push(edgeB.source);

      programNodes.push({
        id: node.id,
        type: "Transformation",
        dataType: "natural",
        operation: OPERATOR_MAP[operator] || "ADD",
        inputs,
      });
    }
  }

  // 3. Encontrar nodo más a la derecha y crear OutputNode
  const rightmost = nodes.reduce((r, n) =>
    n.position.x > r.position.x ? n : r
  );

  const outputId = `output_${rightmost.id}`;
  programNodes.push({
    id: outputId,
    type: "Output",
    dataType: "natural",
    input: rightmost.id,
  });

  // 4. Convertir edges (solo los que van a TransformationNodes)
  // Los DataSource no tienen inputs, así que filtramos edges que apuntan a ellos
  const transformationIds = new Set(
    nodes.filter((n) => n.type === "operator").map((n) => n.id)
  );

  for (const edge of edges) {
    // Solo incluir edges cuyo destino sea un Transformation (operador)
    if (!transformationIds.has(edge.target)) {
      continue;
    }

    const toPort = edge.targetHandle === "b" ? 1 : 0;
    programEdges.push({
      id: edge.id || `${edge.source}_${edge.target}`,
      from: edge.source,
      to: edge.target,
      toPort,
    });
  }

  return {
    metadata: { programId: `canvas-${Date.now()}` },
    graph: { nodes: programNodes, edges: programEdges },
  };
}
