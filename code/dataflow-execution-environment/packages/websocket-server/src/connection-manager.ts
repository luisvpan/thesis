import type { ElysiaWS } from "elysia/ws";

export class ConnectionManager {
  private connections: Set<ElysiaWS<any, any>>;
  private connectionTimestamps: Map<ElysiaWS<any, any>, number>;
  private readonly STALE_TIMEOUT = 5 * 60 * 1000;

  constructor() {
    this.connections = new Set();
    this.connectionTimestamps = new Map();
  }

  addConnection(ws: ElysiaWS<any, any>): void {
    this.connections.add(ws);
    this.connectionTimestamps.set(ws, Date.now());
    this.cleanupStaleConnections();
  }

  removeConnection(ws: ElysiaWS<any, any>): void {
    this.connections.delete(ws);
    this.connectionTimestamps.delete(ws);
  }

  private cleanupStaleConnections(): void {
    const now = Date.now();

    for (const ws of this.connections) {
      const timestamp = this.connectionTimestamps.get(ws)!;
      if (now - timestamp > this.STALE_TIMEOUT || ws.readyState !== 1) {
        this.removeConnection(ws);
      }
    }
  }

  getConnectionCount(): number {
    return this.connections.size;
  }

  getAllConnections(): ElysiaWS<any, any>[] {
    return Array.from(this.connections);
  }
}
