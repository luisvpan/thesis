import { Elysia, t } from "elysia";
import type { ElysiaWS } from "elysia/ws";

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

function broadcastToBrowsers(payload: VisionBroadcastPayload) {
  const msg = JSON.stringify(payload);
  for (const ws of visionSockets) {
    try {
      ws.send(msg);
    } catch {
      visionSockets.delete(ws);
    }
  }
}

/**
 * - `POST /api/v1/vision/ingest`: Python (YOLO) envía el número detectado; se reenvía por WS.
 * - `WS /ws/vision`: el frontend se suscribe para recibir `detectedNumber`.
 */
export const visionModule = new Elysia({ name: "vision" })
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
  );
