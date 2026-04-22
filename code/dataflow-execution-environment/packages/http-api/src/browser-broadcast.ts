import type { ElysiaWS } from "elysia/ws";

/** Clientes navegador en `/ws/touch` (un solo Set para todo el proceso). */
const browserSockets = new Set<ElysiaWS<any, any>>();

export function registerBrowserSocket(ws: ElysiaWS<any, any>) {
  browserSockets.add(ws);
}

export function unregisterBrowserSocket(ws: ElysiaWS<any, any>) {
  browserSockets.delete(ws);
}

export function getBrowserSocketCount(): number {
  return browserSockets.size;
}

/** Reenvío a navegadores en `/ws/touch`. No usar para payloads muy grandes (p. ej. lotes de cartas). */
export function broadcastRawToBrowserClients(msg: string) {
  for (const ws of browserSockets) {
    try {
      ws.send(msg);
    } catch {
      browserSockets.delete(ws);
    }
  }
}
