import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/NodeContext";

/**
 * Firma estable del lienzo para auto-ejecutar solo cuando cambia la topología / datos de entrada,
 * no cuando solo se escriben `value`/`tapError` en cartas `programOutput` tras un run.
 */
export function executionGraphFingerprint(
  nodes: DataflowNode[],
  edges: Edge[]
): string {
  const slimNodes = nodes.map((n) => {
    if (n.type === "number") {
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        value: (n.data as { value?: number }).value,
      };
    }
    if (n.type === "operator") {
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        operator: (n.data as { operator?: string }).operator,
      };
    }
    if (n.type === "programOutput") {
      const d = n.data as {
        pairedAnchorId?: string;
      };
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        pairedAnchorId: d.pairedAnchorId,
      };
    }
    if (n.type === "resultAnchor") {
      const d = n.data as { pairedOutputId?: string };
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        pairedOutputId: d.pairedOutputId,
      };
    }
    return { id: n.id, type: n.type, position: n.position };
  });

  const slimEdges = edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle ?? null,
    targetHandle: e.targetHandle ?? null,
  }));

  return JSON.stringify({ n: slimNodes, e: slimEdges });
}
