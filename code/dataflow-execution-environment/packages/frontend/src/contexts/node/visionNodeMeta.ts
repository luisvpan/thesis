import type { VisionCardItem } from "../VisionContext";

/** Estado de sincronización con visión; `lost`/`stale` atenúan la carta en el lienzo. */
export type VisionNodeStatus = "active" | "lost" | "stale";

export type VisionNodeMeta = {
  trackId?: number;
  visionStatus?: VisionNodeStatus;
  /** Último frame con detección (ms). */
  lastSeenAt?: number;
  /** Primer frame con detección (ms); tras 1.5s se fija la posición en el lienzo. */
  firstSeenAt?: number;
};

export function readVisionMeta(data: unknown): VisionNodeMeta {
  if (!data || typeof data !== "object") return {};
  const d = data as VisionNodeMeta;
  return {
    trackId: d.trackId,
    visionStatus: d.visionStatus,
    lastSeenAt: d.lastSeenAt,
    firstSeenAt: d.firstSeenAt,
  };
}

export function readTrackId(data: unknown): number | undefined {
  return readVisionMeta(data).trackId;
}

export function visionMetaFromCard(
  card: VisionCardItem,
  frameTimeMs: number,
  prevMeta?: VisionNodeMeta
): VisionNodeMeta {
  return {
    trackId: card.trackId,
    visionStatus: card.status === "lost" ? "lost" : "active",
    lastSeenAt: frameTimeMs,
    firstSeenAt: prevMeta?.firstSeenAt ?? frameTimeMs,
  };
}
