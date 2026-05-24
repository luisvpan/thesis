/**
 * Geometría de la zona de arreglo (abrir ↔ cerrar).
 * Mantener alineado con ArrayOpenNode.tsx y ArrayCloseNode.tsx (handles + tamaño de carta).
 */

import type { Edge } from "@xyflow/react";
import type { DataflowNode } from "../contexts/node/types";

/** Tailwind h-52 w-52 → 13rem, asumiendo 1rem = 16px */
export const FLOW_CARD_SIZE = 208;

/** ClickableHandle: h-20 w-20 */
export const FLOW_HANDLE_SIZE = 80;

type Point = { x: number; y: number };

export type AxisAlignedBounds = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

/**
 * Centro del handle zone-out (source Right + translateX(100px)), mismo patrón que SourceFlowNode `out`.
 */
export function getArrayOpenZoneOutCenter(
  node: Pick<{ position: { x: number; y: number } }, "position">
): Point {
  const { x, y } = node.position;
  return {
    x: x + FLOW_CARD_SIZE + 100,
    y: y + FLOW_CARD_SIZE / 2,
  };
}

/** Fracción vertical del handle zone-in (Left). Debe coincidir con ArrayCloseNode.tsx */
export const ARRAY_CLOSE_ZONE_IN_TOP_FRAC = 0.28;

/**
 * Centro del handle zone-in (target Left + top ARRAY_CLOSE_ZONE_IN_TOP_FRAC + translateX(-100px)).
 */
export function getArrayCloseZoneInCenter(
  node: Pick<{ position: { x: number; y: number } }, "position">
): Point {
  const { x, y } = node.position;
  return {
    x: x - 100 + FLOW_HANDLE_SIZE / 2,
    y: y + ARRAY_CLOSE_ZONE_IN_TOP_FRAC * FLOW_CARD_SIZE,
  };
}

/** AABB entre los dos handlers del enlace de zona (orden abrir → cerrar). */
export function getArrayZoneBounds(
  openNode: Pick<{ position: { x: number; y: number } }, "position">,
  closeNode: Pick<{ position: { x: number; y: number } }, "position">
): AxisAlignedBounds {
  const p0 = getArrayOpenZoneOutCenter(openNode);
  const p1 = getArrayCloseZoneInCenter(closeNode);
  return {
    left: Math.min(p0.x, p1.x),
    right: Math.max(p0.x, p1.x),
    top: Math.min(p0.y, p1.y),
    bottom: Math.max(p0.y, p1.y),
  };
}

/** Centro de la carta estándar (208×208) anclada en node.position. */
export function getFlowCardCenter(
  node: Pick<{ position: { x: number; y: number } }, "position">
): Point {
  const { x, y } = node.position;
  return {
    x: x + FLOW_CARD_SIZE / 2,
    y: y + FLOW_CARD_SIZE / 2,
  };
}

/** AABB de la carta estándar (208×208) anclada en node.position. */
export function getFlowCardBounds(
  node: Pick<{ position: { x: number; y: number } }, "position">
): AxisAlignedBounds {
  const { x, y } = node.position;
  return {
    left: x,
    right: x + FLOW_CARD_SIZE,
    top: y,
    bottom: y + FLOW_CARD_SIZE,
  };
}

export function isPointInBoundsInclusive(
  p: Point,
  b: AxisAlignedBounds
): boolean {
  return p.x >= b.left && p.x <= b.right && p.y >= b.top && p.y <= b.bottom;
}

/** Solapa o toca borde (inclusivo): basta con que una parte de la carta entre en la zona. */
export function doAxisAlignedBoundsOverlap(
  a: AxisAlignedBounds,
  b: AxisAlignedBounds
): boolean {
  return (
    a.left <= b.right &&
    a.right >= b.left &&
    a.top <= b.bottom &&
    a.bottom >= b.top
  );
}

export function shouldIncludeNodeInArrayZone(
  n: DataflowNode,
  openId: string,
  closeId: string,
  bounds: AxisAlignedBounds
): boolean {
  if (n.id === openId || n.id === closeId) return false;
  if (n.type === "programOutput") return false;
  if (n.type !== "source" && n.type !== "operator") return false;
  return doAxisAlignedBoundsOverlap(getFlowCardBounds(n), bounds);
}

/** Cartas cuyos handlers se ocultan dentro de la zona (no incluye abrir/cerrar arreglo). */
function isNodeTypeThatHidesHandlesWhenInArrayZone(n: DataflowNode): boolean {
  return (
    n.type === "source" ||
    n.type === "operator" ||
    n.type === "programOutput"
  );
}

/**
 * IDs de nodos cuya carta solapa el AABB de algún par abrir→cerrar conectado por
 * `zone-in`. Unión si hay varias zonas.
 */
/**
 * Nodos fuente/operador dentro de la zona de un `arrayClose`, en el mismo orden
 * que `flowToProgram` (izquierda→derecha, luego arriba→abajo).
 */
export function getOrderedArrayZoneMembers(
  closeNodeId: string,
  nodes: DataflowNode[],
  edges: Edge[]
): DataflowNode[] {
  const zoneEdge = edges.find(
    (e) => e.target === closeNodeId && e.targetHandle === "zone-in"
  );
  if (!zoneEdge) return [];

  const openNode = nodes.find(
    (n) => n.id === zoneEdge.source && n.type === "arrayOpen"
  );
  const closeNode = nodes.find(
    (n) => n.id === closeNodeId && n.type === "arrayClose"
  );
  if (!openNode || !closeNode) return [];

  const bounds = getArrayZoneBounds(openNode, closeNode);
  return nodes
    .filter((n) =>
      shouldIncludeNodeInArrayZone(n, openNode.id, closeNode.id, bounds)
    )
    .sort((a, b) =>
      a.position.x !== b.position.x
        ? a.position.x - b.position.x
        : a.position.y - b.position.y
    );
}

export function computeNodeIdsInsideActiveArrayZones(
  nodes: DataflowNode[],
  edges: Edge[]
): Set<string> {
  const inside = new Set<string>();

  for (const edge of edges) {
    if (edge.targetHandle !== "zone-in") continue;

    const openNode = nodes.find(
      (n) => n.id === edge.source && n.type === "arrayOpen"
    );
    const closeNode = nodes.find(
      (n) => n.id === edge.target && n.type === "arrayClose"
    );
    if (!openNode || !closeNode) continue;

    const bounds = getArrayZoneBounds(openNode, closeNode);

    for (const n of nodes) {
      if (!isNodeTypeThatHidesHandlesWhenInArrayZone(n)) continue;
      if (doAxisAlignedBoundsOverlap(getFlowCardBounds(n), bounds)) {
        inside.add(n.id);
      }
    }
  }

  return inside;
}
