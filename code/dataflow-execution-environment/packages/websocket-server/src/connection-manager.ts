import type { ElysiaWS } from "elysia/ws";

/**
 * Manages WebSocket connections with automatic cleanup of stale connections
 * Tracks connection timestamps and removes connections that have been inactive
 * for longer than the timeout period
 * @class ConnectionManager
 */
export class ConnectionManager {
  private connections: Set<ElysiaWS<any, any>>;
  private connectionTimestamps: Map<ElysiaWS<any, any>, number>;

  /**
   * Timeout in milliseconds after which a connection is considered stale
   * Default: 5 minutes
   * @constant {number}
   */
  private readonly STALE_TIMEOUT = 5 * 60 * 1000;

  /**
   * Creates a new ConnectionManager instance
   * @constructor
   */
  constructor() {
    this.connections = new Set();
    this.connectionTimestamps = new Map();
  }

  /**
   * Adds a new WebSocket connection to the manager
   * @param {ElysiaWS} ws - WebSocket connection instance to add
   * @example
   * connectionManager.addConnection(ws);
   */
  addConnection(ws: ElysiaWS<any, any>): void {
    this.connections.add(ws);
    this.connectionTimestamps.set(ws, Date.now());
    this.cleanupStaleConnections();
  }

  /**
   * Removes a WebSocket connection from the manager
   * @param {ElysiaWS} ws - WebSocket connection instance to remove
   * @example
   * connectionManager.removeConnection(ws);
   */
  removeConnection(ws: ElysiaWS<any, any>): void {
    this.connections.delete(ws);
    this.connectionTimestamps.delete(ws);
  }

  /**
   * Removes stale connections that have been inactive for too long
   * or have a readyState other than OPEN (1)
   * @private
   * @returns {void}
   */
  private cleanupStaleConnections(): void {
    const now = Date.now();

    for (const ws of this.connections) {
      const timestamp = this.connectionTimestamps.get(ws)!;
      if (now - timestamp > this.STALE_TIMEOUT || ws.readyState !== 1) {
        this.removeConnection(ws);
      }
    }
  }

  /**
   * Gets the total number of active connections
   * @returns {number} Count of active WebSocket connections
   * @example
   * const count = connectionManager.getConnectionCount();
   * console.log(`Active connections: ${count}`);
   */
  getConnectionCount(): number {
    return this.connections.size;
  }

  /**
   * Gets all active WebSocket connections
   * @returns {ElysiaWS[]} Array of active WebSocket connection instances
   * @example
   * const connections = connectionManager.getAllConnections();
   * connections.forEach(ws => ws.send(JSON.stringify({ broadcast: true })));
   */
  getAllConnections(): ElysiaWS<any, any>[] {
    return Array.from(this.connections);
  }
}
