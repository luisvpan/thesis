import type { ServerWebSocket } from "elysia";

export class ConnectionManager {
  private connections: Set<ServerWebSocket>;

  constructor() {
    this.connections = new Set();
  }

  addConnection(ws: ServerWebSocket): void {
    this.connections.add(ws);
  }

  removeConnection(ws: ServerWebSocket): void {
    this.connections.delete(ws);
  }

  getConnectionCount(): number {
    return this.connections.size;
  }

  getAllConnections(): ServerWebSocket[] {
    return Array.from(this.connections);
  }
}
