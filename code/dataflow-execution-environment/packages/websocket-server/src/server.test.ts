import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { app } from "./server";
import { IncrementalRuntime } from "@dataflow/runtime";
import type { DataflowProgram, DataflowNode, DataflowEdge } from "@dataflow/shared/types";

describe("WebSocket Server", () => {
  let server: any;
  let wsUrl: string;

  beforeAll(async () => {
    server = app.listen(3001);
    wsUrl = "ws://localhost:3001/live";
    await new Promise(resolve => setTimeout(resolve, 100));
  });

  afterAll(() => {
    server?.stop();
  });

  function createWebSocket(): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => resolve(ws);
      ws.onerror = (error) => reject(error);
    });
  }

  function sendMessage(ws: WebSocket, message: any): void {
    ws.send(JSON.stringify(message));
  }

  function waitForMessage(ws: WebSocket, timeout = 5000): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Timeout")), timeout);
      const handler = (event: MessageEvent) => {
        clearTimeout(timer);
        ws.removeEventListener("message", handler as any);
        try {
          resolve(JSON.parse(event.data));
        } catch {
          resolve(event.data);
        }
      };
      ws.addEventListener("message", handler as any);
    });
  }

  function createSimpleProgram(): DataflowProgram {
    return {
      metadata: { programId: "test-program" },
      graph: {
        nodes: [
          {
            id: "n1",
            type: "DataSource",
            dataType: "natural",
            value: 5
          },
          {
            id: "n2",
            type: "DataSource",
            dataType: "natural",
            value: 3
          },
          {
            id: "add",
            type: "Transformation",
            dataType: "natural",
            operation: "ADD",
            inputs: ["n1", "n2"]
          },
          {
            id: "output",
            type: "Output",
            dataType: "natural",
            input: "add"
          }
        ],
        edges: [
          { id: "e1", from: "n1", to: "add", toPort: 0 },
          { id: "e2", from: "n2", to: "add", toPort: 1 },
          { id: "e3", from: "add", to: "output" }
        ]
      }
    };
  }

  describe("Connection Management", () => {
    it("should accept client connections", async () => {
      const ws = await createWebSocket();
      expect(ws.readyState).toBe(WebSocket.OPEN);
      ws.close();
    });

    it("should handle multiple concurrent connections", async () => {
      const connections = await Promise.all([
        createWebSocket(),
        createWebSocket(),
        createWebSocket(),
        createWebSocket(),
        createWebSocket()
      ]);

      connections.forEach(ws => expect(ws.readyState).toBe(WebSocket.OPEN));
      connections.forEach(ws => ws.close());
    });

    it("should handle client disconnection gracefully", async () => {
      const ws = await createWebSocket();
      ws.close();

      await new Promise(resolve => setTimeout(resolve, 100));

      expect(ws.readyState).toBe(WebSocket.CLOSED);
    });
  });

  describe("validate_program", () => {
    it("should validate a valid program", async () => {
      const ws = await createWebSocket();

      const message = {
        type: "validate_program",
        messageId: "msg_001",
        program: createSimpleProgram()
      };

      sendMessage(ws, message);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("validation_result");
      expect(response.messageId).toBe("msg_001");
      expect(response.errors).toEqual([]);
      expect(response.warnings).toBeDefined();

      ws.close();
    });

    it("should detect validation errors in invalid program", async () => {
      const ws = await createWebSocket();

      const invalidProgram: DataflowProgram = {
        metadata: { programId: "test-program" },
        graph: {
          nodes: [
            {
              id: "n1",
              type: "DataSource",
              dataType: "natural",
              value: 5
            },
            {
              id: "output",
              type: "Output",
              dataType: "natural",
              input: "n2"
            }
          ],
          edges: []
        }
      };

      const message = {
        type: "validate_program",
        messageId: "msg_002",
        program: invalidProgram
      };

      sendMessage(ws, message);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("validation_result");
      expect(response.messageId).toBe("msg_002");
      expect(response.errors.length).toBeGreaterThan(0);

      ws.close();
    });
  });

  describe("evaluate_incremental", () => {
    it("should evaluate a simple program", async () => {
      const ws = await createWebSocket();

      const message = {
        type: "evaluate_incremental",
        messageId: "msg_003",
        program: createSimpleProgram()
      };

      sendMessage(ws, message);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("evaluation_result");
      expect(response.messageId).toBe("msg_003");
      expect(response.nodeStates).toBeDefined();
      expect(response.changedNodes).toBeDefined();

      ws.close();
    });

    it("should handle partial programs with missing inputs", async () => {
      const ws = await createWebSocket();

      const partialProgram: DataflowProgram = {
        metadata: { programId: "test-program" },
        graph: {
          nodes: [
            {
              id: "n1",
              type: "DataSource",
              dataType: "natural",
              value: 5
            },
            {
              id: "add",
              type: "Transformation",
              dataType: "natural",
              operation: "ADD",
              inputs: ["n1", "missing"]
            },
            {
              id: "output",
              type: "Output",
              dataType: "natural",
              input: "add"
            }
          ],
          edges: [
            { id: "e1", from: "n1", to: "add", toPort: 0 }
          ]
        }
      };

      const message = {
        type: "evaluate_incremental",
        messageId: "msg_004",
        program: partialProgram
      };

      sendMessage(ws, message);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("evaluation_result");
      expect(response.nodeStates).toBeDefined();

      ws.close();
    });
  });

  describe("subscribe_node and unsubscribe_node", () => {
    it("should subscribe to a node", async () => {
      const ws = await createWebSocket();

      const subscribeMessage = {
        type: "subscribe_node",
        messageId: "msg_005",
        nodeId: "output"
      };

      sendMessage(ws, subscribeMessage);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("node_state_changed");
      expect(response.messageId).toBe("msg_005");
      expect(response.nodeId).toBe("output");

      ws.close();
    });

    it("should unsubscribe from a node", async () => {
      const ws = await createWebSocket();

      const subscribeMessage = {
        type: "subscribe_node",
        messageId: "msg_006",
        nodeId: "output"
      };

      sendMessage(ws, subscribeMessage);
      await waitForMessage(ws);

      const unsubscribeMessage = {
        type: "unsubscribe_node",
        messageId: "msg_007",
        nodeId: "output"
      };

      sendMessage(ws, unsubscribeMessage);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("node_state_changed");
      expect(response.messageId).toBe("msg_007");
      expect(response.nodeId).toBe("output");

      ws.close();
    });

    it("should handle multiple subscriptions", async () => {
      const ws = await createWebSocket();

      sendMessage(ws, {
        type: "subscribe_node",
        messageId: "msg_008",
        nodeId: "n1"
      });

      await waitForMessage(ws);

      sendMessage(ws, {
        type: "subscribe_node",
        messageId: "msg_009",
        nodeId: "n2"
      });

      const response = await waitForMessage(ws);

      expect(response.type).toBe("node_state_changed");
      expect(response.nodeId).toBe("n2");

      ws.close();
    });
  });

  describe("error handling", () => {
    it("should return error for unknown message type", async () => {
      const ws = await createWebSocket();

      const message = {
        type: "unknown_type",
        messageId: "msg_010"
      };

      sendMessage(ws, message);

      const response = await waitForMessage(ws);

      expect(response.type).toBe("error");
      expect(response.messageId).toBe("msg_010");
      expect(response.code).toBe("INVALID_MESSAGE");

      ws.close();
    });

    it("should handle malformed JSON", async () => {
      const ws = await createWebSocket();

      ws.send("invalid json");

      const response = await waitForMessage(ws);

      expect(response.type).toBe("error");
      expect(response.code).toBe("MESSAGE_PARSE_ERROR");

      ws.close();
    });
  });

  describe("message performance", () => {
    it("should respond to validate_program within 50ms (p95)", async () => {
      const ws = await createWebSocket();

      const message = {
        type: "validate_program",
        messageId: "msg_011",
        program: createSimpleProgram()
      };

      const start = performance.now();
      sendMessage(ws, message);
      await waitForMessage(ws);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(50);

      ws.close();
    });
  });

  describe("concurrent clients", () => {
    it("should handle 5 concurrent WebSocket connections", async () => {
      const clients = await Promise.all([
        createWebSocket(),
        createWebSocket(),
        createWebSocket(),
        createWebSocket(),
        createWebSocket()
      ]);

      const messages = clients.map((ws, index) => ({
        type: "validate_program",
        messageId: `msg_${index}`,
        program: createSimpleProgram()
      }));

      messages.forEach((msg, index) => sendMessage(clients[index], msg));

      const responses = await Promise.all(
        clients.map(ws => waitForMessage(ws))
      );

      responses.forEach((response, index) => {
        expect(response.type).toBe("validation_result");
        expect(response.messageId).toBe(`msg_${index}`);
      });

      clients.forEach(ws => ws.close());
    });
  });
});
