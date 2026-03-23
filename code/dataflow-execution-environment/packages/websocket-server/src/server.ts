import { Elysia } from "elysia";
import type { ElysiaWS } from "elysia/ws";
import { DagValidator } from "@dataflow/compiler";
import { IncrementalRuntime } from "@dataflow/runtime";
import type { DataflowProgram } from "@dataflow/shared/types";
import { ConnectionManager } from "./connection-manager";
import { createRateLimiter } from "@dataflow/shared/security/rate-limiter";

interface WebSocketMessage {
  type: string;
  messageId?: string;
  [key: string]: unknown;
}

interface ValidateProgramMessage extends WebSocketMessage {
  type: "validate_program";
  program: DataflowProgram;
}

interface EvaluateIncrementalMessage extends WebSocketMessage {
  type: "evaluate_incremental";
  program: DataflowProgram;
}

interface SubscribeNodeMessage extends WebSocketMessage {
  type: "subscribe_node";
  nodeId: string;
}

interface UnsubscribeNodeMessage extends WebSocketMessage {
  type: "unsubscribe_node";
  nodeId: string;
}

interface ClientData {
  subscribedNodes: Set<string>;
}

const validator = new DagValidator();
const connectionManager = new ConnectionManager();
const incrementalRuntime = new IncrementalRuntime();

const messageRateLimiter = createRateLimiter({
  windowMs: 60000,
  maxRequests: 300
});

export const app = new Elysia()
  .ws("/live", {
    open(ws: ElysiaWS<any, any>) {
      const currentConnections = connectionManager.getConnectionCount();
      const MAX_CONNECTIONS = 100;
      
      if (currentConnections >= MAX_CONNECTIONS) {
        console.log("Connection rejected: too many connections");
        const errorResponse = JSON.stringify({
          type: "error",
          code: "CONNECTION_LIMIT_EXCEEDED",
          message: "Too many connections"
        });
        ws.send(errorResponse);
        ws.close();
        return;
      }
      
      console.log("Client connected");
      ws.data = {
        subscribedNodes: new Set()
      };
      connectionManager.addConnection(ws);
    },
 
    message: async (ws: ElysiaWS<any, any>, message: any) => {
      let msg: WebSocketMessage;

      if (typeof message === "object") {
        console.log("Received message as already-parsed object");
        msg = message as WebSocketMessage;
      } else {
        const messageStr = typeof message === "string" ? message : message.toString("utf-8");
        console.log("Received message type:", typeof message);
        console.log("Received message:", messageStr);
        try {
          msg = JSON.parse(messageStr) as WebSocketMessage;
        } catch (parseError) {
          const errorResponse = JSON.stringify({
            type: "error",
            messageId: undefined,
            code: "MESSAGE_PARSE_ERROR",
            message: parseError instanceof Error ? parseError.message : "Unknown error"
          });
          console.log("Sending parse error:", errorResponse);
          ws.send(errorResponse);
          return;
        }
      }

      const ip = ws.data.remoteAddress || 'unknown';
      const rateCheck = messageRateLimiter.check(ip);
      
      if (!rateCheck.allowed) {
        const errorResponse = JSON.stringify({
          type: "error",
          messageId: msg.messageId,
          code: "RATE_LIMIT_EXCEEDED",
          message: `Rate limit exceeded. Try again after ${Math.ceil((rateCheck.resetTime - Date.now()) / 1000)}s`
        });
        console.log("Sending rate limit error:", errorResponse);
        ws.send(errorResponse);
        return;
      }

      try {
        switch (msg.type) {
          case "validate_program": {
            const validateMsg = msg as ValidateProgramMessage;
            const result = validator.validateProgram(validateMsg.program);
            const response = JSON.stringify({
              type: "validation_result",
              messageId: validateMsg.messageId,
              errors: result.errors,
              warnings: result.warnings
            });
            console.log("Sending validation_result:", response);
            ws.send(response);
            break;
          }

          case "evaluate_incremental": {
            const evalMsg = msg as EvaluateIncrementalMessage;
            incrementalRuntime.loadProgram(evalMsg.program);
 
            const TIMEOUT_MS = 5000;
            const evaluationPromise = new Promise((resolve) => {
              const evaluation = incrementalRuntime.evaluatePartial(0);
              resolve(evaluation);
            });
            
            const timeoutPromise = new Promise((_, reject) => {
              setTimeout(() => reject(new Error('Evaluation timeout')), TIMEOUT_MS);
            });
            
            const evaluation = await Promise.race([evaluationPromise, timeoutPromise]) as any;
 
            const response = JSON.stringify({
              type: "evaluation_result",
              messageId: evalMsg.messageId,
              nodeStates: Object.fromEntries(evaluation.nodeStates),
              changedNodes: evaluation.changedNodes
            });
            console.log("Sending evaluation_result:", response);
            ws.send(response);
            break;
          }

          case "subscribe_node": {
            const subMsg = msg as SubscribeNodeMessage;
            const { nodeId } = subMsg;

            ws.data?.subscribedNodes?.add(nodeId);

            incrementalRuntime.subscribe(nodeId, (state) => {
              if (ws.readyState === 1) {
                ws.send(JSON.stringify({
                  type: "node_state_changed",
                  nodeId,
                  state
                }));
              }
            });

            const response = JSON.stringify({
              type: "node_state_changed",
              nodeId,
              messageId: subMsg.messageId
            });
            console.log("Sending subscribe confirmation:", response);
            ws.send(response);
            break;
          }

          case "unsubscribe_node": {
            const unsubMsg = msg as UnsubscribeNodeMessage;
            const { nodeId } = unsubMsg;

            ws.data?.subscribedNodes?.delete(nodeId);
            incrementalRuntime.unsubscribe(nodeId, () => {});

            const response = JSON.stringify({
              type: "node_state_changed",
              nodeId,
              messageId: unsubMsg.messageId
            });
            console.log("Sending unsubscribe confirmation:", response);
            ws.send(response);
            break;
          }

          default:
            const response = JSON.stringify({
              type: "error",
              messageId: msg.messageId,
              code: "INVALID_MESSAGE",
              message: "Unknown message type"
            });
            console.log("Sending error:", response);
            ws.send(response);
        }
      } catch (error) {
        let errorCode = "MESSAGE_ERROR";
        let errorMessage = error instanceof Error ? error.message : "Unknown error";
        
        if (error instanceof Error && error.message.includes('Evaluation timeout')) {
          errorCode = "EVALUATION_TIMEOUT";
          errorMessage = "Evaluation took too long";
        }
        
        const errorResponse = JSON.stringify({
          type: "error",
          messageId: undefined,
          code: errorCode,
          message: errorMessage
        });
        console.log("Sending error:", errorResponse);
        ws.send(errorResponse);
      }
    },

    close(ws: ElysiaWS<any, any>) {
      console.log("Client disconnected");

      for (const nodeId of ws.data?.subscribedNodes || []) {
        incrementalRuntime.unsubscribe(nodeId, () => {});
      }

      connectionManager.removeConnection(ws);
    }
  });

export type App = typeof app;
