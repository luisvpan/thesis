import type { ElysiaWS } from "elysia/ws";
import type { NodeState } from "@dataflow/runtime";

type SubscriptionCallback = (state: NodeState) => void;

interface Subscription {
  ws: ElysiaWS<any, any>;
  callback: SubscriptionCallback;
}

export class SubscriptionManager {
  private subscriptions: Map<string, Set<Subscription>>;

  constructor() {
    this.subscriptions = new Map();
  }

  subscribe(nodeId: string, ws: ElysiaWS<any, any>, callback: SubscriptionCallback): void {
    if (!this.subscriptions.has(nodeId)) {
      this.subscriptions.set(nodeId, new Set());
    }

    this.subscriptions.get(nodeId)!.add({ ws, callback });
  }

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

  getSubscriptionCount(nodeId: string): number {
    return this.subscriptions.get(nodeId)?.size || 0;
  }
}
