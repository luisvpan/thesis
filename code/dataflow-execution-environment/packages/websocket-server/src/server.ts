import { Elysia } from "elysia";
import type { ElysiaWS } from "elysia/ws";
import { DagValidator } from "@dataflow/compiler";
import { IncrementalRuntime } from "@dataflow/runtime";
import type { DataflowProgram } from "@dataflow/shared/types";
import { ConnectionManager } from "./connection-manager";
import { createRateLimiter } from "@dataflow/shared/security/rate-limiter";

/**
 * Client runtime data structure for WebSocket connections
 * @interface ClientRuntimeData
 * @property {Set<string>} subscribedNodes - Set of node IDs this client is subscribed to
 * @property {IncrementalRuntime} runtime - The incremental runtime instance for this client
 * @property {string} remoteAddress - Client's remote address for rate limiting
 * @property {Map<string, function>} callbacks - Map of node IDs to their change callbacks
 */
interface ClientRuntimeData {
  subscribedNodes: Set<string>;
  runtime: IncrementalRuntime;
  remoteAddress: string;
  callbacks: Map<string, (state: any) => void>;
}

let messageIdCounter = 0;

/**
 * Generates a unique message ID for WebSocket message tracking
 * @returns {string} Unique message identifier
 * @example
 * const msgId = generateMessageId(); // "msg_0", "msg_1", ...
 */
function generateMessageId(): string {
  return `msg_${messageIdCounter++}`;
}

let connectionIdCounter = 0;

/**
 * Map of connection IDs to their associated client runtime data
 * @type {Map<string, ClientRuntimeData>}
 */
const clientRuntimeMap = new Map<string, ClientRuntimeData>();

const validator = new DagValidator();
const connectionManager = new ConnectionManager();

/**
 * Rate limiter for WebSocket messages (300 messages per 60 seconds per IP)
 * @type {import("@dataflow/shared/security/rate-limiter").RateLimiter}
 */
const messageRateLimiter = createRateLimiter({
  windowMs: 60000,
  maxRequests: 300
});

export const app = new Elysia()
  .ws("/live", {
    /**
     * WebSocket connection open event handler
     * Initializes client connection, creates runtime instance, and adds to connection manager
     * @event open
     * @param {ElysiaWS} ws - WebSocket connection instance
     * @throws {Error} Closes connection if maximum connections limit is reached
     */
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
      
      const remoteAddress = ws.remoteAddress || 'unknown';
      
      console.log("Client connected");
      const connectionId = `conn_${connectionIdCounter++}`;
      const clientRuntime = new IncrementalRuntime();
      const clientData: ClientRuntimeData = {
        subscribedNodes: new Set(),
        runtime: clientRuntime,
        remoteAddress,
        callbacks: new Map()
      };
      clientRuntimeMap.set(connectionId, clientData);
      ws.data.connectionId = connectionId;
      ws.data.subscribedNodes = new Set();
      connectionManager.addConnection(ws);
    },
 
    /**
     * WebSocket message event handler
     * Processes incoming messages and routes to appropriate handlers based on message type
     * @event message
     * @param {ElysiaWS} ws - WebSocket connection instance
     * @param {any} message - Incoming message (string, buffer, or parsed object)
     */
    message: async (ws: ElysiaWS<any, any>, message: any) => {
      let msg: any;

      if (typeof message === "object") {
        console.log("Received message as already-parsed object");
        msg = message;
      } else {
        const messageStr = typeof message === "string" ? message : message.toString("utf-8");
        console.log("Received message type:", typeof message);
        console.log("Received message:", messageStr);
        try {
          msg = JSON.parse(messageStr);
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

      const connectionId = ws.data?.connectionId as string;
      const clientData = connectionId ? clientRuntimeMap.get(connectionId) : undefined;
      const ip = clientData?.remoteAddress || 'unknown';
      const rateCheck = messageRateLimiter.check(ip);
      
      if (!rateCheck.allowed) {
        const errorResponse = JSON.stringify({
          type: "error",
          messageId: msg.messageId || generateMessageId(),
          code: "RATE_LIMIT_EXCEEDED",
          message: `Rate limit exceeded. Try again after ${Math.ceil((rateCheck.resetTime - Date.now()) / 1000)}s`
        });
        console.log("Sending rate limit error:", errorResponse);
        ws.send(errorResponse);
        return;
      }

      try {
        switch (msg.type) {
          /**
           * Validates a dataflow program without executing it
           * @messageType validate_program
           * @param {Object} msg - Validation request message
           * @param {DataflowProgram} msg.program - The program to validate
           * @param {string} [msg.messageId] - Optional message ID for request tracking
           * @returns {Object} Validation result with errors and warnings
           */
          case "validate_program": {
            const result = validator.validateProgram(msg.program);
            const response = JSON.stringify({
              type: "validation_result",
              messageId: msg.messageId || generateMessageId(),
              errors: result.errors,
              warnings: result.warnings
            });
            console.log("Sending validation_result:", response);
            ws.send(response);
            break;
          }

          /**
           * Evaluates a program incrementally using the client's incremental runtime
           * @messageType evaluate_incremental
           * @param {Object} msg - Evaluation request message
           * @param {DataflowProgram} msg.program - The program to evaluate
           * @param {string} [msg.messageId] - Optional message ID for request tracking
           * @returns {Object} Evaluation result with node states and changed nodes
           * @throws {Error} If runtime not found, evaluation times out, or other errors occur
           */
          case "evaluate_incremental": {
            const connectionId = ws.data?.connectionId as string;
            const clientData = connectionId ? clientRuntimeMap.get(connectionId) : undefined;
            const clientRuntime = clientData?.runtime;
            if (!clientRuntime) {
              const errorResponse = JSON.stringify({
                type: "error",
                messageId: msg.messageId || generateMessageId(),
                code: "RUNTIME_NOT_FOUND",
                message: "Client runtime not found"
              });
              ws.send(errorResponse);
              break;
            }

            clientRuntime.loadProgram(msg.program);

            const TIMEOUT_MS = 5000;
            let timeoutTriggered = false;
            const evaluationPromise = new Promise((resolve, reject) => {
              setImmediate(() => {
                if (timeoutTriggered) {
                  reject(new Error('Evaluation timeout'));
                  return;
                }
                try {
                  const evaluation = clientRuntime.evaluatePartial(0);
                  resolve(evaluation);
                } catch (error) {
                  reject(error);
                }
              });
            });
            
            const timeoutPromise = new Promise((_, reject) => {
              setTimeout(() => {
                timeoutTriggered = true;
                reject(new Error('Evaluation timeout'));
              }, TIMEOUT_MS);
            });
            
            const evaluation = await Promise.race([evaluationPromise, timeoutPromise]) as any;

            const response = JSON.stringify({
              type: "evaluation_result",
              messageId: msg.messageId || generateMessageId(),
              nodeStates: Object.fromEntries(evaluation.nodeStates),
              changedNodes: evaluation.changedNodes
            });
            console.log("Sending evaluation_result:", response);
            ws.send(response);
            break;
          }

          /**
           * Subscribes to state changes for a specific node
           * @messageType subscribe_node
           * @param {Object} msg - Subscription request message
           * @param {string} msg.nodeId - The ID of the node to subscribe to
           * @param {string} [msg.messageId] - Optional message ID for request tracking
           * @returns {Object} Confirmation of successful subscription
           * @throws {Error} If runtime not found
           */
          case "subscribe_node": {
            const { nodeId } = msg;
            const connectionId = ws.data?.connectionId as string;
            const clientData = connectionId ? clientRuntimeMap.get(connectionId) : undefined;
            const clientRuntime = clientData?.runtime;
            if (!clientRuntime) {
              const errorResponse = JSON.stringify({
                type: "error",
                messageId: msg.messageId || generateMessageId(),
                code: "RUNTIME_NOT_FOUND",
                message: "Client runtime not found"
              });
              ws.send(errorResponse);
              break;
            }

            clientData?.subscribedNodes?.add(nodeId);

            const callback = (state: any) => {
              if (ws.readyState === 1) {
                ws.send(JSON.stringify({
                  type: "node_state_changed",
                  nodeId,
                  messageId: msg.messageId || generateMessageId(),
                  state
                }));
              }
            };

            clientRuntime.subscribe(nodeId, callback);
            clientData?.callbacks?.set(nodeId, callback);

            const response = JSON.stringify({
              type: "node_state_changed",
              nodeId,
              messageId: msg.messageId || generateMessageId()
            });
            console.log("Sending subscribe confirmation:", response);
            ws.send(response);
            break;
          }

          /**
           * Unsubscribes from state changes for a specific node
           * @messageType unsubscribe_node
           * @param {Object} msg - Unsubscription request message
           * @param {string} msg.nodeId - The ID of the node to unsubscribe from
           * @param {string} [msg.messageId] - Optional message ID for request tracking
           * @returns {Object} Confirmation of successful unsubscription
           * @throws {Error} If runtime not found
           */
          case "unsubscribe_node": {
            const { nodeId } = msg;
            const connectionId = ws.data?.connectionId as string;
            const clientData = connectionId ? clientRuntimeMap.get(connectionId) : undefined;
            const clientRuntime = clientData?.runtime;
            if (!clientRuntime) {
              const errorResponse = JSON.stringify({
                type: "error",
                messageId: msg.messageId || generateMessageId(),
                code: "RUNTIME_NOT_FOUND",
                message: "Client runtime not found"
              });
              ws.send(errorResponse);
              break;
            }

            const callback = clientData?.callbacks?.get(nodeId);
            if (callback) {
              clientRuntime.unsubscribe(nodeId, callback);
              clientData?.callbacks?.delete(nodeId);
            }

            clientData?.subscribedNodes?.delete(nodeId);

            const response = JSON.stringify({
              type: "node_state_changed",
              nodeId,
              messageId: msg.messageId || generateMessageId()
            });
            console.log("Sending unsubscribe confirmation:", response);
            ws.send(response);
            break;
          }

          default:
            const response = JSON.stringify({
              type: "error",
              messageId: msg.messageId || generateMessageId(),
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
          messageId: msg.messageId || generateMessageId(),
          code: errorCode,
          message: errorMessage
        });
        console.log("Sending error:", errorResponse);
        ws.send(errorResponse);
      }
    },

    /**
     * WebSocket connection close event handler
     * Cleans up client resources including subscriptions and removes from connection manager
     * @event close
     * @param {ElysiaWS} ws - WebSocket connection instance
     */
    close(ws: ElysiaWS<any, any>) {
      console.log("Client disconnected");

      const connectionId = ws.data?.connectionId as string;
      const clientData = connectionId ? clientRuntimeMap.get(connectionId) : undefined;
      const clientRuntime = clientData?.runtime;
      if (clientRuntime && clientData?.callbacks) {
        for (const [nodeId, callback] of clientData.callbacks.entries()) {
          clientRuntime.unsubscribe(nodeId, callback);
        }
        clientData.callbacks.clear();
      }

      if (connectionId) {
        clientRuntimeMap.delete(connectionId);
      }
      connectionManager.removeConnection(ws);
    }
  });

export type App = typeof app;
