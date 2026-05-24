import type { Node } from "@xyflow/react";
import type { VisionNodeMeta, VisionNodeStatus } from "./visionNodeMeta";

/** Cartas visibles y con tracking activo — encima del resto. */
export const VISION_ACTIVE_Z_INDEX = 2;
/** Tracking perdido u ocluido (ByteTrack `lost`) o fuera de frame (`stale`). */
export const VISION_DIMMED_Z_INDEX = 0;
export const VISION_DIMMED_OPACITY = 0.45;

export function isVisionTrackDegraded(
  status?: VisionNodeStatus
): boolean {
  return status === "lost" || status === "stale";
}

export function visionNodeZIndex(status?: VisionNodeStatus): number {
  return isVisionTrackDegraded(status)
    ? VISION_DIMMED_Z_INDEX
    : VISION_ACTIVE_Z_INDEX;
}

export function visionNodePresentationFields(
  meta: VisionNodeMeta
): Pick<Node, "zIndex" | "style"> {
  const degraded = isVisionTrackDegraded(meta.visionStatus);
  return {
    zIndex: visionNodeZIndex(meta.visionStatus),
    style: degraded ? { opacity: VISION_DIMMED_OPACITY } : undefined,
  };
}

/** Aplica z-index, opacidad y flag draggable de carta en el lienzo. */
export function withVisionNodeChrome<T extends Node>(
  node: T,
  meta: VisionNodeMeta,
  nodesDraggable: boolean
): T {
  const { zIndex, style } = visionNodePresentationFields(meta);
  return {
    ...node,
    zIndex,
    style: { ...(node.style ?? {}), ...style },
    draggable: nodesDraggable,
  };
}
