import type { VisionCardItem } from "../VisionContext";
import { VISION_POSITION_LOCK_MS } from "./constants";
import { visionToFlowPosition } from "./helpers";
import { readVisionMeta } from "./visionNodeMeta";
import type { DataflowNode } from "./types";

/** Posición en el lienzo: sigue visión al inicio; tras {@link VISION_POSITION_LOCK_MS} queda fija. */
export function resolveVisionNodePosition(
  nodeId: string,
  card: VisionCardItem,
  rect: DOMRectReadOnly,
  frameTimeMs: number,
  prevCards: DataflowNode[]
): { x: number; y: number } {
  const prevNode = prevCards.find((n) => n.id === nodeId);
  const visionPos = visionToFlowPosition(card.position, rect);
  if (!prevNode) return visionPos;

  const { firstSeenAt } = readVisionMeta(prevNode.data);
  const firstSeen = firstSeenAt ?? frameTimeMs;
  if (frameTimeMs - firstSeen >= VISION_POSITION_LOCK_MS) {
    return prevNode.position;
  }
  return visionPos;
}
