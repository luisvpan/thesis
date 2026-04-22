import { Elysia, t } from "elysia";

import {
  broadcastRawToBrowserClients,
  getBrowserSocketCount,
  registerBrowserSocket,
  unregisterBrowserSocket,
} from "./browser-broadcast";

const touchEventSchema = t.Object({
  type: t.Literal("touch"),
  position: t.Object({ x: t.Number(), y: t.Number() }),
  timestamp: t.String(),
});

function broadcastToBrowsers(payload: object) {
  const msg = JSON.stringify(payload);
  broadcastRawToBrowserClients(msg);
}

export const touchModule = new Elysia({ name: "touch" })
  .ws("/ws/touch", {
    open(ws) {
      registerBrowserSocket(ws);
      console.log("[touch] browser connected:", getBrowserSocketCount());
    },
    close(ws) {
      unregisterBrowserSocket(ws);
      console.log("[touch] browser disconnected:", getBrowserSocketCount());
    },
  })
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
        `[touch] (${message.position.x}, ${message.position.y}) → relay ok`,
      );
    },
  });
