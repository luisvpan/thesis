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
  /** Marca de tiempo al pasar a `lost` (evita parpadeo de opacidad). */
  lostSinceMs?: number;
};

/** Campos de visión que no deben invalidar el hash del programa ni re-ejecutar. */
export const VISION_META_KEYS: (keyof VisionNodeMeta)[] = [
  "trackId",
  "visionStatus",
  "lastSeenAt",
  "firstSeenAt",
  "lostSinceMs",
];

/** Copia de `data` sin metadatos de tracking (para hashes / comparación estable). */
export function dataWithoutVisionMeta(data: unknown): Record<string, unknown> {
  if (!data || typeof data !== "object") return {};
  const out = { ...(data as Record<string, unknown>) };
  for (const key of VISION_META_KEYS) {
    delete out[key];
  }
  return out;
}

/** Tiempo en estado `lost` antes de atenuar la carta (ms). */
export const VISION_LOST_DIM_AFTER_MS = 400;

export function readVisionMeta(data: unknown): VisionNodeMeta {
  if (!data || typeof data !== "object") return {};
  const d = data as VisionNodeMeta;
  return {
    trackId: d.trackId,
    visionStatus: d.visionStatus,
    lastSeenAt: d.lastSeenAt,
    firstSeenAt: d.firstSeenAt,
    lostSinceMs: d.lostSinceMs,
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
  const rawLost = card.status === "lost";
  let visionStatus: VisionNodeStatus = rawLost ? "lost" : "active";
  let lostSinceMs = prevMeta?.lostSinceMs;

  if (rawLost) {
    lostSinceMs = prevMeta?.lostSinceMs ?? frameTimeMs;
    const wasActive =
      !prevMeta ||
      prevMeta.visionStatus === "active" ||
      prevMeta.visionStatus === undefined;
    if (wasActive && frameTimeMs - lostSinceMs < VISION_LOST_DIM_AFTER_MS) {
      visionStatus = "active";
    }
  } else {
    lostSinceMs = undefined;
  }

  return {
    trackId: card.trackId,
    visionStatus,
    lastSeenAt: frameTimeMs,
    firstSeenAt: prevMeta?.firstSeenAt ?? frameTimeMs,
    lostSinceMs,
  };
}
