/**
 * @deprecated Este archivo usa el formato antiguo DataflowProgram para http-api.
 * Usar flowToProgram.ts en su lugar, que genera el formato Program del interpreter
 * para ejecución client-side.
 *
 * Serializa el programa visual (ReactFlow) al formato DataflowProgram del backend.
 * Los nodos `resultAnchor` y cualquier arista que los toque son solo UI y no se envían.
 * Los nodos `programOutput` se serializan como DataSource numéricos (valor tras ejecutar en cliente).
 */

import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/NodeContext";
import type { ProgramOutputFlowNodeData } from "@/components/dataflow";

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

  const evalNodes = nodes.filter(
    (n) => n.type === "number" || n.type === "operator"
  );

  if (evalNodes.length === 0) {
    return {
      metadata: { programId: `canvas-${Date.now()}` },
      graph: { nodes: [], edges: [] },
    };
  }

  // 1. DataSource: números y carta de resultado (frontend → número opaco para el runtime)
  for (const node of nodes) {
    if (node.type === "resultAnchor") continue;

    if (node.type === "number") {
      const value = (node.data as { value?: number }).value ?? 0;
      programNodes.push({
        id: node.id,
        type: "DataSource",
        dataType: "natural",
        value: { kind: "natural", value },
      });
      continue;
    }

    if (node.type === "programOutput") {
      const value =
        (node.data as ProgramOutputFlowNodeData).value ?? 0;
      programNodes.push({
        id: node.id,
        type: "DataSource",
        dataType: "natural",
        value: { kind: "natural", value },
      });
      continue;
    }
  }

  // 2. Transformaciones
  for (const node of nodes) {
    if (node.type !== "operator") continue;

    const operator = (node.data as { operator?: string }).operator ?? "adicion";

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

  // 3. Output: solo nodos evaluables (no cartas solo-frontend)
  const rightmost = evalNodes.reduce((r, n) =>
    n.position.x > r.position.x ? n : r
  );

  const outputId = `output_${rightmost.id}`;
  programNodes.push({
    id: outputId,
    type: "Output",
    dataType: "natural",
    input: rightmost.id,
  });

  const transformationIds = new Set(
    nodes.filter((n) => n.type === "operator").map((n) => n.id)
  );

  for (const edge of edges) {
    if (!transformationIds.has(edge.target)) {
      continue;
    }

    const src = nodes.find((n) => n.id === edge.source);
    const tgt = nodes.find((n) => n.id === edge.target);
    if (src?.type === "resultAnchor" || tgt?.type === "resultAnchor") {
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
