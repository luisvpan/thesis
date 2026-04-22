import { Elysia, t } from "elysia";
import type { ElysiaWS } from "elysia/ws";
import type { DataflowProgram } from "@dataflow/shared/types";
import { executeDataflowProgram } from "./execute-dataflow";

/**
 * WebSocket para ejecutar programas dataflow sin POST.
 * Mensajes entrantes (JSON): `{ type: "execute", program, requestId? }` o `{ type: "ping" }`.
 * Respuestas: `{ type: "executeResult", ... }` o `{ type: "pong" }`.
 */
export const dataflowWsModule = new Elysia({ name: "dataflow-ws" }).ws(
  "/ws/dataflow",
  {
    body: t.Any(),
    open(ws: ElysiaWS<any, any>) {
      console.log("[dataflow-ws] cliente conectado");
      ws.send(
        JSON.stringify({
          type: "ready",
          t: Date.now(),
        }),
      );
    },
    close() {
      console.log("[dataflow-ws] cliente desconectado");
    },
    message(ws, raw) {
      const body = raw as { type?: string; requestId?: string; program?: DataflowProgram };
      if (body.type === "ping") {
        ws.send(JSON.stringify({ type: "pong", t: Date.now() }));
        return;
      }

      if (body.type !== "execute") return;

      const requestId = body.requestId ?? undefined;
      if (!body.program) {
        ws.send(
          JSON.stringify({
            type: "executeResult",
            requestId,
            success: false,
            error: "Falta el programa (program)",
          }),
        );
        return;
      }

      try {
        const program = body.program as DataflowProgram;
        const result = executeDataflowProgram(program);

        if (!result.success) {
          ws.send(
            JSON.stringify({
              type: "executeResult",
              requestId,
              success: false,
              programId: result.programId,
              errors: result.errors,
            }),
          );
          return;
        }

        ws.send(
          JSON.stringify({
            type: "executeResult",
            requestId,
            success: true,
            programId: result.programId,
            outputs: result.outputs,
            totalTimeMs: result.totalTimeMs,
          }),
        );
      } catch (err) {
        ws.send(
          JSON.stringify({
            type: "executeResult",
            requestId,
            success: false,
            error: err instanceof Error ? err.message : String(err),
          }),
        );
      }
    },
  },
);
