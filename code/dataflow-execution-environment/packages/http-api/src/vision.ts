import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { Elysia, t } from "elysia";
import type { ElysiaWS } from "elysia/ws";

/** Misma convención que `CameraConfig`: `rgb_resolution` es `[height, width]` en píxeles. */
export type RgbResolution = { width: number; height: number };

const DEFAULT_RGB: RgbResolution = { width: 1920, height: 1080 };

/**
 * Ruta a `session.json` del CV system. Por defecto: `code/cv-system/config/session.json`
 * relativo al monorepo (desde `packages/http-api/src`).
 */
export function resolveSessionJsonPath(): string {
  const env = process.env.CV_SESSION_JSON_PATH ?? process.env.SESSION_JSON_PATH;
  if (env && env.length > 0) return env;
  return join(import.meta.dir, "../../../../cv-system/config/session.json");
}

/**
 * Lee `camera.rgb_resolution` del session del CV. Si no existe o falla, devuelve DEFAULT_RGB.
 */
export function readRgbResolutionFromSession(): { resolution: RgbResolution; ok: boolean } {
  const path = resolveSessionJsonPath();
  if (!existsSync(path)) {
    return { resolution: { ...DEFAULT_RGB }, ok: false };
  }
  try {
    const raw = readFileSync(path, "utf-8");
    const j = JSON.parse(raw) as { camera?: { rgb_resolution?: [number, number] } };
    const rgb = j.camera?.rgb_resolution;
    if (!Array.isArray(rgb) || rgb.length < 2) {
      return { resolution: { ...DEFAULT_RGB }, ok: false };
    }
    const [h, w] = rgb;
    if (typeof h !== "number" || typeof w !== "number" || h <= 0 || w <= 0) {
      return { resolution: { ...DEFAULT_RGB }, ok: false };
    }
    return { resolution: { width: w, height: h }, ok: true };
  } catch {
    return { resolution: { ...DEFAULT_RGB }, ok: false };
  }
}

/** Clientes WebSocket conectados a `/ws/vision` (navegador). */
const visionSockets = new Set<ElysiaWS<any, any>>();

export type VisionBroadcastPayload = {
  type: "detectedNumber";
  /** Índice de clase YOLO (0..nc-1) */
  classId: number;
  /** Nombre de clase según data.yaml */
  label: string;
  /** Valor 1..9 solo si la clase es one..nine; si no aplica, omitido */
  number?: number;
  confidence?: number;
  /** Centro del bbox normalizado al frame de la cámara (0..1), para React Flow */
  position?: { x: number; y: number };
  t: number;
};

/** Una carta detectada en vista de proyector (YOLO); posición normalizada 0-1. */
export type VisionCardItem = {
  classId: number;
  label: string;
  confidence: number;
  trackId?: number;  // Persistent tracking ID from YOLO tracker
  status?: "active" | "lost";
  position: { x: number; y: number };
  bbox?: { x1: number; y1: number; x2: number; y2: number };
};

export type VisionCardDetectionsPayload = {
  type: "cardDetections";
  cards: VisionCardItem[];
  t: number;
};

function broadcastRaw(msg: string) {
  for (const ws of visionSockets) {
    try {
      ws.send(msg);
    } catch {
      visionSockets.delete(ws);
    }
  }
}

function broadcastToBrowsers(payload: VisionBroadcastPayload) {
  broadcastRaw(JSON.stringify(payload));
}

/**
 * - `GET /api/v1/vision/projector-resolution`: lee `rgb_resolution` de session.json (CV) para React Flow.
 * - `POST /api/v1/vision/ingest`: Python (YOLO) envía el número detectado; se reenvía por WS.
 * - `POST /api/v1/vision/cards`: Python envía todas las cartas del frame (posiciones en vista proyector).
 * - `WS /ws/vision`: el frontend se suscribe para recibir `detectedNumber` y `cardDetections`.
 */
export const visionModule = new Elysia({ name: "vision" })
  .get("/api/v1/vision/projector-resolution", () => {
    const { resolution, ok } = readRgbResolutionFromSession();
    return {
      rgbResolution: resolution,
      sessionPath: resolveSessionJsonPath(),
      fromSessionFile: ok,
    };
  })
  .ws("/ws/vision", {
    open(ws: ElysiaWS<any, any>) {
      visionSockets.add(ws);
      console.log("[vision] browser ws connected, clients:", visionSockets.size);
    },
    close(ws: ElysiaWS<any, any>) {
      visionSockets.delete(ws);
      console.log("[vision] browser ws closed, clients:", visionSockets.size);
    },
  })
  .post(
    "/api/v1/vision/ingest",
    ({ body }) => {
      const payload: VisionBroadcastPayload = {
        type: "detectedNumber",
        classId: body.classId,
        label: body.label,
        number: body.number,
        confidence: body.confidence,
        position: body.position,
        t: Date.now(),
      };
      broadcastToBrowsers(payload);
      return { ok: true as const, forwarded: visionSockets.size };
    },
    {
      body: t.Object({
        classId: t.Number(),
        label: t.String(),
        number: t.Optional(t.Number()),
        confidence: t.Optional(t.Number()),
        position: t.Optional(
          t.Object({
            x: t.Number(),
            y: t.Number(),
          }),
        ),
      }),
    },
  )
  .post(
    "/api/v1/vision/cards",
    ({ body }) => {
      const payload: VisionCardDetectionsPayload = {
        type: "cardDetections",
        cards: body.cards,
        t: body.t ?? Date.now(),
      };
      broadcastRaw(JSON.stringify(payload));
      return { ok: true as const, forwarded: visionSockets.size, count: body.cards.length };
    },
    {
      body: t.Object({
        cards: t.Array(
          t.Object({
            classId: t.Number(),
            label: t.String(),
            confidence: t.Number(),
            trackId: t.Optional(t.Number()),
            status: t.Optional(t.Union([t.Literal("active"), t.Literal("lost")])),
            position: t.Object({
              x: t.Number(),
              y: t.Number(),
            }),
            bbox: t.Optional(
              t.Object({
                x1: t.Number(),
                y1: t.Number(),
                x2: t.Number(),
                y2: t.Number(),
              }),
            ),
          }),
        ),
        t: t.Optional(t.Number()),
      }),
    },
  );
