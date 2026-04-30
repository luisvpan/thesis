import type {
  OperatorFlowNodeData,
  ProgramOutputFlowNodeData,
  SourceFlowNodeData,
} from "@/components/dataflow";
import {
  VISION_FLOW_MIN_SIZE,
  VISION_NODE_HALF_H,
  VISION_NODE_HALF_W,
} from "./constants";
import type { DataflowNode } from "./types";

/**
 * Convierte coordenadas normalizadas (0-1) a coordenadas del ReactFlow.
 * El CV envía posiciones normalizadas respecto al viewport del proyector.
 */
export function visionToFlowPosition(
  pos: { x: number; y: number },
  flowRect: Pick<DOMRectReadOnly, "left" | "top" | "width" | "height">
): { x: number; y: number } {
  if (
    flowRect.width < VISION_FLOW_MIN_SIZE ||
    flowRect.height < VISION_FLOW_MIN_SIZE
  ) {
    return { x: 0, y: 0 };
  }

  const viewportX = pos.x * window.innerWidth;
  const viewportY = pos.y * window.innerHeight;

  let x = viewportX - flowRect.left - VISION_NODE_HALF_W;
  let y = viewportY - flowRect.top - VISION_NODE_HALF_H;

  const maxX = Math.max(0, flowRect.width - 2 * VISION_NODE_HALF_W);
  const maxY = Math.max(0, flowRect.height - 2 * VISION_NODE_HALF_H);
  x = Math.max(0, Math.min(x, maxX));
  y = Math.max(0, Math.min(y, maxY));

  return { x, y };
}

export function getNodeValue(node: DataflowNode | null | undefined): number | undefined {
  if (!node?.data) return undefined;
  if (node.type === "programOutput") {
    const v = (node.data as ProgramOutputFlowNodeData).value;
    return typeof v === "number" ? v : undefined;
  }
  if (node.type === "source") {
    const d = node.data as SourceFlowNodeData;
    return d.variant === "number" ? d.value : undefined;
  }
  const d = node.data as OperatorFlowNodeData;
  return d.result;
}

export function getRightmostEvaluableNode(nodes: DataflowNode[]): DataflowNode | null {
  const evalNodes = nodes.filter(
    (n): n is Extract<DataflowNode, { type: "source" | "operator" }> =>
      n.type === "source" || n.type === "operator"
  );
  if (evalNodes.length === 0) return null;
  return evalNodes.reduce((rightmost, node) =>
    node.position.x > rightmost.position.x ? node : rightmost
  );
}

/** Slug válido para el DSL a partir del trackId (fallback por índice). */
export function toValidSlug(trackId: number | undefined, fallbackIndex: number): string {
  if (trackId !== undefined && trackId >= 0) {
    return `card_${trackId}`;
  }
  return `card_${fallbackIndex}`;
}
