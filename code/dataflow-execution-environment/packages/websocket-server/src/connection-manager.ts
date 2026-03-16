import type { ElysiaWS } from "elysia/ws";

export class ConnectionManager {
  private connections: Set<ElysiaWS<any, any>>;

  constructor() {
    this.connections = new Set();
  }

  addConnection(ws: ElysiaWS<any, any>): void {
    this.connections.add(ws);
  }

  removeConnection(ws: ElysiaWS<any, any>): void {
    this.connections.delete(ws);
  }

  getConnectionCount(): number {
    return this.connections.size;
  }

  getAllConnections(): ElysiaWS<any, any>[] {
    return Array.from(this.connections);
  }
}
