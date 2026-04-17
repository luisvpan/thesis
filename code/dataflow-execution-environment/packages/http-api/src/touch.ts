import { Elysia, t } from "elysia";
import type { ElysiaWS } from "elysia/ws";

const touchEventSchema = t.Object({
  type: t.Literal("touch"),
  position: t.Object({ x: t.Number(), y: t.Number() }),
  timestamp: t.String(),
});

// Browser clients subscribed to touch events
const browserSockets = new Set<ElysiaWS<any, any>>();

function broadcastToBrowsers(payload: object) {
  const msg = JSON.stringify(payload);
  for (const ws of browserSockets) {
    try {
      ws.send(msg);
    } catch {
      browserSockets.delete(ws);
    }
  }
}

export const touchModule = new Elysia({ name: "touch" })
  // Browser clients connect here
  .ws("/ws/touch", {
    open(ws) {
      browserSockets.add(ws);
      console.log("[touch] browser connected:", browserSockets.size);
    },
    close(ws) {
      browserSockets.delete(ws);
      console.log("[touch] browser disconnected:", browserSockets.size);
    },
  })
  // CV System connects here
  .ws("/live", {
    body: touchEventSchema,
    open(ws) {
      console.log("[touch] CV system connected");
    },
    close(ws) {
      console.log("[touch] CV system disconnected");
    },
    message(ws, message) {
      const payload = { ...message, t: Date.now() };
      broadcastToBrowsers(payload);
      console.log(
        `[touch] (${message.position.x}, ${message.position.y}) → ${browserSockets.size} browsers`
      );
    },
  });
