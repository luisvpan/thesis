import type { ElysiaWS } from "elysia/ws";
import type { NodeState } from "@dataflow/runtime";

/**
 * Callback function type for node state change notifications
 * @callback SubscriptionCallback
 * @param {NodeState} state - The updated node state
 */
type SubscriptionCallback = (state: NodeState) => void;

/**
 * Represents a single subscription to a node's state changes
 * @interface Subscription
 * @property {ElysiaWS} ws - WebSocket connection for the subscriber
 * @property {SubscriptionCallback} callback - Callback function invoked on state changes
 */
interface Subscription {
  ws: ElysiaWS<any, any>;
  callback: SubscriptionCallback;
}

/**
 * Manages WebSocket subscriptions to node state changes
 * Tracks which connections are subscribed to which nodes and notifies
 * subscribers when node states change
 * @class SubscriptionManager
 */
export class SubscriptionManager {
  private subscriptions: Map<string, Set<Subscription>>;

  /**
   * Creates a new SubscriptionManager instance
   * @constructor
   */
  constructor() {
    this.subscriptions = new Map();
  }

  /**
   * Subscribes a WebSocket connection to state changes for a specific node
   * @param {string} nodeId - The ID of the node to subscribe to
   * @param {ElysiaWS} ws - WebSocket connection to subscribe
   * @param {SubscriptionCallback} callback - Callback function invoked when node state changes
   * @example
   * subscriptionManager.subscribe("node_123", ws, (state) => {
   *   console.log("Node state changed:", state);
   * });
   */
  subscribe(nodeId: string, ws: ElysiaWS<any, any>, callback: SubscriptionCallback): void {
    if (!this.subscriptions.has(nodeId)) {
      this.subscriptions.set(nodeId, new Set());
    }

    this.subscriptions.get(nodeId)!.add({ ws, callback });
  }

  /**
   * Unsubscribes a WebSocket connection from state changes for a specific node
   * @param {string} nodeId - The ID of the node to unsubscribe from
   * @param {ElysiaWS} ws - WebSocket connection to unsubscribe
   * @example
   * subscriptionManager.unsubscribe("node_123", ws);
   */
  unsubscribe(nodeId: string, ws: ElysiaWS<any, any>): void {
    const subs = this.subscriptions.get(nodeId);
    if (!subs) return;

    for (const sub of subs) {
      if (sub.ws === ws) {
        subs.delete(sub);
        break;
      }
    }

    if (subs.size === 0) {
      this.subscriptions.delete(nodeId);
    }
  }

  /**
   * Removes all subscriptions for a specific WebSocket connection
   * Useful when a connection closes and all its subscriptions should be cleaned up
   * @param {ElysiaWS} ws - WebSocket connection whose subscriptions should be removed
   * @example
   * subscriptionManager.removeAllForConnection(ws);
   */
  removeAllForConnection(ws: ElysiaWS<any, any>): void {
    for (const [nodeId, subs] of this.subscriptions.entries()) {
      for (const sub of subs) {
        if (sub.ws === ws) {
          subs.delete(sub);
        }
      }
      if (subs.size === 0) {
        this.subscriptions.delete(nodeId);
      }
    }
  }

  /**
   * Notifies all subscribers of a state change for a specific node
   * Errors in individual callbacks are caught and logged but don't affect other subscribers
   * @param {string} nodeId - The ID of the node whose state changed
   * @param {NodeState} state - The new node state
   * @example
   * subscriptionManager.notify("node_123", { value: 42, timestamp: Date.now() });
   */
  notify(nodeId: string, state: NodeState): void {
    const subs = this.subscriptions.get(nodeId);
    if (!subs) return;

    for (const sub of subs) {
      try {
        sub.callback(state);
      } catch (error) {
        console.error(`Error notifying subscriber for node ${nodeId}:`, error);
      }
    }
  }

  /**
   * Gets the number of active subscriptions for a specific node
   * @param {string} nodeId - The ID of the node to check
   * @returns {number} Count of active subscriptions for the node
   * @example
   * const count = subscriptionManager.getSubscriptionCount("node_123");
   * console.log(`Node has ${count} subscribers`);
   */
  getSubscriptionCount(nodeId: string): number {
    return this.subscriptions.get(nodeId)?.size || 0;
  }
}
