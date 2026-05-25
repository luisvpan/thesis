import type { Edge } from "@xyflow/react";
import type { DataflowNode, PortIdentifier } from "@/contexts/node/types";
import { isOrderOperatorType, type OperatorType } from "@/types/card-types";
import { acceptsConnection } from "./handle-kinds";
import type { PortKindInfo } from "@/contexts/node/types";
import type { OperatorFlowNodeData } from "./OperatorFlowNode";

export type PortRef = PortIdentifier;

export type ConnectionContext = {
  nodes: DataflowNode[];
  edges: Edge[];
  getPortKindInfo: (nodeId: string, handleId: string) => PortKindInfo | undefined;
};

function nodeType(nodes: DataflowNode[], nodeId: string): DataflowNode["type"] | undefined {
  return nodes.find((n) => n.id === nodeId)?.type;
}

/** True if this port already has an edge attached. */
export function isPortOccupied(edges: Edge[], port: PortRef): boolean {
  if (port.handleType === "source") {
    return edges.some(
      (e) => e.source === port.nodeId && (e.sourceHandle ?? null) === port.handleId
    );
  }
  return edges.some(
    (e) => e.target === port.nodeId && (e.targetHandle ?? null) === port.handleId
  );
}

/** arrayOpen → arrayClose with targetHandle zone-in exists for this close node. */
export function isArrayClosePairedWithOpen(
  closeNodeId: string,
  edges: Edge[],
  nodes: DataflowNode[]
): boolean {
  return edges.some((e) => {
    if (e.target !== closeNodeId || e.targetHandle !== "zone-in") return false;
    const src = nodes.find((n) => n.id === e.source);
    return src?.type === "arrayOpen" && e.sourceHandle === "zone-out";
  });
}

function operatorType(nodes: DataflowNode[], nodeId: string): OperatorType | undefined {
  const node = nodes.find((n) => n.id === nodeId);
  if (node?.type !== "operator") return undefined;
  return (node.data as OperatorFlowNodeData | undefined)?.operator;
}

function isOperatorInputHandle(
  nodes: DataflowNode[],
  operatorNodeId: string,
  handleId: string
): boolean {
  const op = operatorType(nodes, operatorNodeId);
  if (op && isOrderOperatorType(op)) {
    return handleId === "a";
  }
  return handleId === "a" || handleId === "b";
}

function isOrderOperatorTarget(
  nodes: DataflowNode[],
  targetNodeId: string,
  targetHandle: string
): boolean {
  const op = operatorType(nodes, targetNodeId);
  return op != null && isOrderOperatorType(op) && targetHandle === "a";
}

/**
 * Structural connection rules (a)–(d) from the plan, before handle-kind checks.
 */
export function canConnectStructurally(
  source: PortRef,
  target: PortRef,
  ctx: Pick<ConnectionContext, "nodes" | "edges">
): { ok: boolean; reason?: string } {
  if (source.handleType !== "source" || target.handleType !== "target") {
    return { ok: false, reason: "port-type-mismatch" };
  }
  if (source.nodeId === target.nodeId) {
    return { ok: false, reason: "same-node" };
  }

  const srcType = nodeType(ctx.nodes, source.nodeId);
  const tgtType = nodeType(ctx.nodes, target.nodeId);
  const srcHandle = source.handleId;
  const tgtHandle = target.handleId;

  if (!srcType || !tgtType) {
    return { ok: false, reason: "unknown-node" };
  }

  // (a) Entrada → solo operador (a|b), excepto ordenar (solo grupos)
  if (srcType === "source") {
    if (srcHandle !== "out") return { ok: false, reason: "source-handle" };
    if (tgtType === "operator" && isOrderOperatorTarget(ctx.nodes, target.nodeId, tgtHandle)) {
      return { ok: false, reason: "order-requires-group" };
    }
    if (
      tgtType === "operator" &&
      isOperatorInputHandle(ctx.nodes, target.nodeId, tgtHandle)
    ) {
      return { ok: true };
    }
    return { ok: false, reason: "source-target" };
  }

  // Abrir → cerrar zone-in únicamente
  if (srcType === "arrayOpen") {
    if (srcHandle === "zone-out" && tgtType === "arrayClose" && tgtHandle === "zone-in") {
      return { ok: true };
    }
    return { ok: false, reason: "array-open-target" };
  }

  // Cerrar out → salida in | operador (si pareja abrir)
  if (srcType === "arrayClose" && srcHandle === "out") {
    if (tgtType === "programOutput" && tgtHandle === "in") {
      return { ok: true };
    }
    if (tgtType === "operator" && isOperatorInputHandle(ctx.nodes, target.nodeId, tgtHandle)) {
      if (isArrayClosePairedWithOpen(source.nodeId, ctx.edges, ctx.nodes)) {
        return { ok: true };
      }
      return { ok: false, reason: "array-close-unpaired" };
    }
    return { ok: false, reason: "array-close-out-target" };
  }

  // Operador out → salida in | operador a|b
  if (srcType === "operator" && srcHandle === "out") {
    if (tgtType === "programOutput" && tgtHandle === "in") {
      return { ok: true };
    }
    if (tgtType === "operator" && isOperatorInputHandle(ctx.nodes, target.nodeId, tgtHandle)) {
      if (isOrderOperatorTarget(ctx.nodes, target.nodeId, tgtHandle)) {
        if (srcType === "operator") return { ok: true };
        return { ok: false, reason: "order-requires-group" };
      }
      return { ok: true };
    }
    return { ok: false, reason: "operator-out-target" };
  }

  // Salida out → solo operador a|b (no a ordenar: solo grupos)
  if (srcType === "programOutput" && srcHandle === "out") {
    if (tgtType === "operator" && isOrderOperatorTarget(ctx.nodes, target.nodeId, tgtHandle)) {
      return { ok: false, reason: "order-requires-group" };
    }
    if (tgtType === "operator" && isOperatorInputHandle(ctx.nodes, target.nodeId, tgtHandle)) {
      return { ok: true };
    }
    return { ok: false, reason: "sink-out-target" };
  }

  return { ok: false, reason: "structural" };
}

/** Full validation: occupied ports, structure, and handle kinds. */
export function canConnectPorts(
  source: PortRef,
  target: PortRef,
  ctx: ConnectionContext
): { ok: boolean; reason?: string } {
  if (isPortOccupied(ctx.edges, source) || isPortOccupied(ctx.edges, target)) {
    return { ok: false, reason: "occupied" };
  }

  const structural = canConnectStructurally(source, target, ctx);
  if (!structural.ok) {
    return structural;
  }

  const sourceInfo = ctx.getPortKindInfo(source.nodeId, source.handleId);
  const targetInfo = ctx.getPortKindInfo(target.nodeId, target.handleId);
  const sourceKind = sourceInfo?.produces ?? "any";
  const targetAccepts = targetInfo?.accepts ?? ["any"];

  if (!acceptsConnection(sourceKind, targetAccepts)) {
    return { ok: false, reason: "handle-kind" };
  }

  return { ok: true };
}

/** Whether two ports could connect if user selects them (ignores occupancy of endpoints being evaluated for highlight). */
export function wouldPortsConnect(
  first: PortRef,
  second: PortRef,
  ctx: ConnectionContext
): boolean {
  if (first.handleType === second.handleType) return false;
  if (first.nodeId === second.nodeId) return false;

  const source = first.handleType === "source" ? first : second;
  const target = first.handleType === "target" ? first : second;

  if (isPortOccupied(ctx.edges, source) || isPortOccupied(ctx.edges, target)) {
    return false;
  }

  return canConnectPorts(source, target, ctx).ok;
}

export type PortHighlightState =
  | "idle"
  | "connected"
  | "selected"
  | "compatible"
  | "incompatible";

export function getPortHighlightState(
  port: PortRef,
  ctx: ConnectionContext & { selectedPort: PortRef | null }
): PortHighlightState {
  if (isPortOccupied(ctx.edges, port)) {
    return "connected";
  }

  if (
    ctx.selectedPort &&
    ctx.selectedPort.nodeId === port.nodeId &&
    ctx.selectedPort.handleId === port.handleId &&
    ctx.selectedPort.handleType === port.handleType
  ) {
    return "selected";
  }

  if (!ctx.selectedPort) {
    return "idle";
  }

  if (ctx.selectedPort.handleType === port.handleType) {
    return "idle";
  }

  if (wouldPortsConnect(ctx.selectedPort, port, ctx)) {
    return "compatible";
  }

  return "incompatible";
}
