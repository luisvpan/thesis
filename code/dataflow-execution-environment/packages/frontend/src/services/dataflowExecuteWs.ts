/**
 * Ejecución dataflow vía WebSocket (`/ws/dataflow`) en lugar de POST.
 */

import {
  serializeProgram,
  type DataflowProgram,
} from "@/utils/serializeProgram";
import type { DataflowNode } from "@/contexts/NodeContext";
import type { Edge } from "@xyflow/react";
import type { ExecuteResult } from "@/services/executeProgram";

export function getDataflowWsUrl(): string {
  if (typeof import.meta.env.VITE_DATAFLOW_WS_URL === "string" && import.meta.env.VITE_DATAFLOW_WS_URL) {
    return import.meta.env.VITE_DATAFLOW_WS_URL;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/dataflow`;
}

type Pending = {
  resolve: (r: ExecuteResult) => void;
};

let socket: WebSocket | null = null;
let connectPromise: Promise<WebSocket> | null = null;
let seq = 0;

const pending = new Map<string, Pending>();

const connectionListeners = new Set<(connected: boolean) => void>();

function notifyConnection(connected: boolean) {
  for (const fn of connectionListeners) {
    try {
      fn(connected);
    } catch {
      /* ignore */
    }
  }
}

function rejectAllPending(message: string) {
  for (const [, w] of pending) {
    w.resolve({ success: false, error: message });
  }
  pending.clear();
}

function handleMessage(raw: string) {
  try {
    const data = JSON.parse(raw) as Record<string, unknown>;
    if (data.type === "ready" || data.type === "pong") {
      return;
    }
    if (data.type !== "executeResult") return;

    const requestId = data.requestId as string | undefined;
    if (requestId == null || requestId === "") {
      return;
    }

    const waiter = pending.get(requestId);
    if (!waiter) return;
    pending.delete(requestId);

    if (data.success === false) {
      const errMsg =
        Array.isArray(data.errors) && data.errors.length > 0
          ? String((data.errors as { message?: string }[])[0]?.message ?? data.errors)
          : (data.error as string) ||
            (data.errors != null ? JSON.stringify(data.errors) : "Error de ejecución");
      waiter.resolve({ success: false, error: errMsg });
      return;
    }

    const output = (data.outputs as unknown[])?.[0];
    let result: number | undefined;
    if (output === undefined || output === null) {
      waiter.resolve({ success: false, error: "Sin resultado" });
      return;
    }
    if (typeof output === "object" && output !== null && "value" in output) {
      result = (output as { value: number }).value;
    } else if (typeof output === "number") {
      result = output;
    } else {
      result = Number(output);
    }

    waiter.resolve({ success: true, result });
  } catch (e) {
    console.warn("[dataflowExecuteWs] mensaje inválido", e);
  }
}

/** Abre el WebSocket de ejecución si hace falta (útil para el indicador de estado). */
export function ensureDataflowExecuteSocket(): Promise<WebSocket> {
  if (socket?.readyState === WebSocket.OPEN) {
    return Promise.resolve(socket);
  }
  if (connectPromise) return connectPromise;

  connectPromise = new Promise((resolve, reject) => {
    const url = getDataflowWsUrl();
    const ws = new WebSocket(url);

    ws.onmessage = (ev: MessageEvent) => {
      handleMessage(String(ev.data));
    };

    ws.onopen = () => {
      socket = ws;
      notifyConnection(true);
      resolve(ws);
      connectPromise = null;
    };

    ws.onclose = () => {
      socket = null;
      connectPromise = null;
      notifyConnection(false);
      rejectAllPending("WebSocket cerrado");
    };

    ws.onerror = () => {
      connectPromise = null;
      notifyConnection(false);
      reject(new Error("WebSocket de ejecución: error de conexión"));
    };
  });

  return connectPromise;
}

export function subscribeDataflowWsStatus(cb: (connected: boolean) => void): () => void {
  connectionListeners.add(cb);
  cb(socket != null && socket.readyState === WebSocket.OPEN);
  return () => {
    connectionListeners.delete(cb);
  };
}

export async function executeProgramViaWs(
  nodes: DataflowNode[],
  edges: Edge[],
  programOverride?: DataflowProgram
): Promise<ExecuteResult> {
  if (nodes.length === 0 && !programOverride) {
    return { success: false, error: "No hay nodos para ejecutar" };
  }

  const program = programOverride ?? serializeProgram(nodes, edges);
  const requestId = `r_${++seq}_${Date.now()}`;

  try {
    const ws = await ensureDataflowExecuteSocket();

    return await new Promise<ExecuteResult>((resolve) => {
      const timeout = window.setTimeout(() => {
        if (!pending.has(requestId)) return;
        pending.delete(requestId);
        resolve({ success: false, error: "Tiempo de espera de ejecución agotado" });
      }, 8000);

      pending.set(requestId, {
        resolve: (r) => {
          window.clearTimeout(timeout);
          resolve(r);
        },
      });

      try {
        ws.send(
          JSON.stringify({
            type: "execute",
            requestId,
            program,
          })
        );
      } catch (e) {
        pending.delete(requestId);
        window.clearTimeout(timeout);
        resolve({
          success: false,
          error: e instanceof Error ? e.message : "Error al enviar",
        });
      }
    });
  } catch (e) {
    return {
      success: false,
      error: e instanceof Error ? e.message : "Error de conexión WebSocket",
    };
  }
}
