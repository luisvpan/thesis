import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "@/contexts/NodeContext";
import type { ProgramOutputFlowNodeData } from "@/components/dataflow/ProgramOutputFlowNode";

/**
 * Nodos insertados por visión (`card_*` / `card_uva_*`) mueven posición cada frame.
 * Si incluimos `position` en la firma, el debounce de auto-ejecutar dispara en bucle y puede tumbar la UI.
 */
function includePositionInFingerprint(nodeId: string): boolean {
  return !nodeId.startsWith("card_");
}

/**
 * Firma estable del lienzo para auto-ejecutar solo cuando cambia la topología / datos de entrada,
 * no cuando solo se escriben `value`/`tapError` en cartas `programOutput` tras un run.
 */
export function executionGraphFingerprint(
  nodes: DataflowNode[],
  edges: Edge[]
): string {
  const slimNodes = nodes.map((n) => {
    const pos =
      includePositionInFingerprint(n.id) && n.position != null
        ? n.position
        : undefined;

    if (n.type === "number") {
      return {
        id: n.id,
        type: n.type,
        ...(pos !== undefined ? { position: pos } : {}),
        value: (n.data as { value?: number }).value,
      };
    }
    if (n.type === "operator") {
      return {
        id: n.id,
        type: n.type,
        ...(pos !== undefined ? { position: pos } : {}),
        operator: (n.data as { operator?: string }).operator,
      };
    }
    if (n.type === "programOutput") {
      const d = n.data as ProgramOutputFlowNodeData;
      return {
        id: n.id,
        type: n.type,
        ...(pos !== undefined ? { position: pos } : {}),
        pairedAnchorId: d.pairedAnchorId,
        displayMode: d.displayMode,
      };
    }
    if (n.type === "resultAnchor") {
      const d = n.data as { pairedOutputId?: string };
      return {
        id: n.id,
        type: n.type,
        ...(pos !== undefined ? { position: pos } : {}),
        pairedOutputId: d.pairedOutputId,
      };
    }
    if (n.type === "deckProp") {
      return {
        id: n.id,
        type: n.type,
        ...(pos !== undefined ? { position: pos } : {}),
        variant: n.data.variant,
      };
    }
    const _exhaustive: never = n;
    void _exhaustive;
    throw new Error("executionGraphFingerprint: tipo de nodo no contemplado");
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
