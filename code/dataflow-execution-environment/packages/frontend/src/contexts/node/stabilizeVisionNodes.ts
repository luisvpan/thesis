import type { DataflowNode } from "./types";
import { readVisionMeta, type VisionNodeMeta } from "./visionNodeMeta";

const POSITION_EPS_PX = 1;

/** Campos que cambian cada frame pero no alteran tipo/valor de la carta. */
const EPHEMERAL_META_KEYS: (keyof VisionNodeMeta)[] = [
  "lastSeenAt",
  "firstSeenAt",
  "visionStatus",
  "lostSinceMs",
  "trackId",
];

function stripEphemeralMeta(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== "object") return {};
  const out = { ...(data as Record<string, unknown>) };
  for (const key of EPHEMERAL_META_KEYS) {
    delete out[key];
  }
  return out;
}

function nodePayloadEqual(a: DataflowNode, b: DataflowNode): boolean {
  if (a.type !== b.type) return false;
  return (
    JSON.stringify(stripEphemeralMeta(a.data)) ===
    JSON.stringify(stripEphemeralMeta(b.data))
  );
}

function positionStable(
  a: { x: number; y: number },
  b: { x: number; y: number }
): boolean {
  return (
    Math.abs(a.x - b.x) <= POSITION_EPS_PX &&
    Math.abs(a.y - b.y) <= POSITION_EPS_PX
  );
}

function visionPresentationEqual(a: DataflowNode, b: DataflowNode): boolean {
  return (
    a.zIndex === b.zIndex &&
    JSON.stringify(a.style) === JSON.stringify(b.style) &&
    readVisionMeta(a.data).visionStatus === readVisionMeta(b.data).visionStatus
  );
}

function canReusePrevNode(prev: DataflowNode, next: DataflowNode): boolean {
  return (
    prev.id === next.id &&
    positionStable(prev.position, next.position) &&
    nodePayloadEqual(prev, next)
  );
}

/** Aplica metadatos de visión y estilo de oclusión sin recrear el nodo entero. */
function mergeEphemeralVisionFields(
  prev: DataflowNode,
  next: DataflowNode
): DataflowNode {
  if (visionPresentationEqual(prev, next)) {
    const prevMeta = readVisionMeta(prev.data);
    const nextMeta = readVisionMeta(next.data);
    if (prevMeta.lastSeenAt === nextMeta.lastSeenAt) {
      return prev;
    }
  }

  const nextMeta = readVisionMeta(next.data);
  return {
    ...prev,
    zIndex: next.zIndex,
    style: next.style,
    data: {
      ...(prev.data as Record<string, unknown>),
      ...nextMeta,
    },
  };
}

/**
 * Reutiliza nodos estables entre frames para reducir parpadeo, pero siempre
 * propaga `lastSeenAt` y el estilo de oclusión (`active` / `lost` / `stale`).
 */
export function stabilizeVisionNodeList(
  prev: DataflowNode[],
  next: DataflowNode[]
): DataflowNode[] {
  if (prev.length !== next.length) return next;

  const prevById = new Map(prev.map((n) => [n.id, n]));
  let allSameRef = true;

  const merged = next.map((n) => {
    const p = prevById.get(n.id);
    if (!p) {
      allSameRef = false;
      return n;
    }
    if (!canReusePrevNode(p, n)) {
      allSameRef = false;
      return n;
    }
    const patched = mergeEphemeralVisionFields(p, n);
    if (patched !== p) allSameRef = false;
    return patched;
  });

  if (allSameRef && merged.every((n) => prevById.has(n.id))) {
    return prev;
  }
  return merged;
}
