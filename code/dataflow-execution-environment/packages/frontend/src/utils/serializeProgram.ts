/**
 * Serializa el programa visual (ReactFlow) al formato DataflowProgram del backend.
 * Los nodos `resultAnchor` no forman parte del grafo semántico (salvo cadena de uvas en `serializeProgramUpToOperator`, donde `programOutput` es DataSource constante).
 * `serializeProgram` usa el nodo evaluable más a la derecha como salida (flujos sin uvas).
 * `serializeProgramUpToOperator` calcula el programa hasta un operador concreto (cartas uva).
 */

import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/NodeContext";

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

const OPERATOR_MAP: Record<string, string> = {
  adicion: "ADD",
  sustraccion: "SUBTRACT",
  multiplicacion: "MULTIPLY",
  division: "DIVIDE",
};

/**
 * Recorre hacia atrás desde un operador e incluye números, operadores y cartas `programOutput`.
 * Una carta de resultado previa cuenta como fuente constante (`data.value`), sin seguir aristas
 * entrantes al interior del otro tap (cadena de uvas).
 */
export function collectUpstreamEvalIds(
  terminalOperatorId: string,
  nodes: DataflowNode[],
  edges: Edge[]
): Set<string> {
  const reachable = new Set<string>();
  const stack = [terminalOperatorId];

  while (stack.length > 0) {
    const id = stack.pop()!;
    if (reachable.has(id)) continue;
    reachable.add(id);

    const current = nodes.find((n) => n.id === id);
    /** Hoja: resultado ya materializado por un tap anterior; no subir más el grafo por aquí */
    if (current?.type === "programOutput") {
      continue;
    }

    for (const e of edges) {
      if (e.target !== id) continue;
      const srcNode = nodes.find((n) => n.id === e.source);
      if (!srcNode) continue;
      if (srcNode.type === "resultAnchor") continue;
      if (!reachable.has(e.source)) {
        stack.push(e.source);
      }
    }
  }

  return reachable;
}

/**
 * Programa que termina en `terminalOperatorId`: salida del lenguaje = resultado de ese operador.
 * Incluye `programOutput` como DataSource constante cuando alimentan un operador (cadena de uvas).
 */
export function serializeProgramUpToOperator(
  terminalOperatorId: string,
  nodes: DataflowNode[],
  edges: Edge[]
): DataflowProgram | null {
  const terminal = nodes.find(
    (n) => n.id === terminalOperatorId && n.type === "operator"
  );
  if (!terminal) return null;

  const reachable = collectUpstreamEvalIds(terminalOperatorId, nodes, edges);

  const evalIds = [...reachable].filter((id) => {
    const n = nodes.find((x) => x.id === id);
    return (
      n?.type === "number" ||
      n?.type === "operator" ||
      n?.type === "programOutput"
    );
  });

  if (evalIds.length === 0) {
    return {
      metadata: { programId: `tap-${terminalOperatorId}-${Date.now()}` },
      graph: { nodes: [], edges: [] },
    };
  }

  const programNodes: ProgramNode[] = [];
  const programEdges: DataflowEdge[] = [];

  for (const id of evalIds) {
    const node = nodes.find((n) => n.id === id);
    if (!node) continue;

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
      const value = (node.data as { value?: number }).value ?? 0;
      programNodes.push({
        id: node.id,
        type: "DataSource",
        dataType: "natural",
        value: { kind: "natural", value },
      });
    }
  }

  const transformationIds = new Set(
    evalIds.filter((id) => nodes.find((n) => n.id === id)?.type === "operator")
  );

  for (const id of transformationIds) {
    const node = nodes.find((n) => n.id === id && n.type === "operator");
    if (!node || node.type !== "operator") continue;

    const operator =
      (node.data as { operator?: string }).operator ?? "adicion";
    const inputEdges = edges.filter(
      (e) => e.target === node.id && reachable.has(e.source)
    );
    const edgeA = inputEdges.find((e) => e.targetHandle === "a");
    const edgeB = inputEdges.find((e) => e.targetHandle === "b");
    const inputs: string[] = [];
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

  const outputId = `output_${terminalOperatorId}`;
  programNodes.push({
    id: outputId,
    type: "Output",
    dataType: "natural",
    input: terminalOperatorId,
  });

  for (const edge of edges) {
    if (!transformationIds.has(edge.target)) continue;
    if (!reachable.has(edge.source) || !reachable.has(edge.target)) continue;

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
    metadata: { programId: `tap-${terminalOperatorId}-${Date.now()}` },
    graph: { nodes: programNodes, edges: programEdges },
  };
}

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

  for (const node of nodes) {
    if (node.type === "resultAnchor" || node.type === "programOutput") continue;

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
    if (
      src?.type === "resultAnchor" ||
      tgt?.type === "resultAnchor" ||
      src?.type === "programOutput" ||
      tgt?.type === "programOutput"
    ) {
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
