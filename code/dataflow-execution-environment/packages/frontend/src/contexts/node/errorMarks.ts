/**
 * Qué papel juega cada carta en el error de una salida, para poder señalarla en
 * el lienzo: la que lo causó, aquella donde ocurrió, la salida que se apagó, y
 * el resto del flujo, que solo se atenúa.
 */

import type { Edge } from "@xyflow/react";
import type { OutputErrorInfo } from "@/services/executeProgram";
import { canvasNodeIdOf } from "@/utils/describeNode";
import { safeJsonStringify } from "@/utils/jsonReplacer";
import type { DataflowNode } from "./types";

export type NodeErrorRole = "cause" | "where" | "sink" | "flow";

export type NodeErrorMark = {
  role: NodeErrorRole;
  /** Encabezado del texto de apoyo; el papel `flow` no lo tiene, solo se atenúa. */
  text?: string;
  /** Detalle y solución probable, para la carta de salida. */
  details?: string[];
};

/** Si una carta cae en varios papeles, manda el más accionable. */
const PRIORITY: Record<NodeErrorRole, number> = { cause: 4, where: 3, sink: 2, flow: 1 };

function keep(current: NodeErrorMark | undefined, next: NodeErrorMark): NodeErrorMark {
  if (!current) return next;
  return PRIORITY[next.role] > PRIORITY[current.role] ? next : current;
}

/** Todo lo que alimenta a una salida, siguiendo las aristas hacia atrás. */
function flowOf(outputNodeId: string, edges: Edge[]): Set<string> {
  const reached = new Set<string>();
  const pending = [outputNodeId];

  while (pending.length > 0) {
    const id = pending.pop()!;
    if (reached.has(id)) continue;
    reached.add(id);
    for (const edge of edges) {
      if (edge.target === id) pending.push(edge.source);
    }
  }

  return reached;
}

export function computeErrorMarks(
  edges: Edge[],
  errorsByOutput: Map<string, OutputErrorInfo[]>
): Map<string, NodeErrorMark> {
  const marks = new Map<string, NodeErrorMark>();
  const put = (nodeId: string, mark: NodeErrorMark) =>
    marks.set(nodeId, keep(marks.get(nodeId), mark));

  for (const [outputId, errors] of errorsByOutput) {
    if (errors.length === 0) continue;

    // El flujo entero se atenúa, y encima van las marcas de los implicados.
    for (const nodeId of flowOf(outputId, edges)) put(nodeId, { role: "flow" });

    put(outputId, {
      role: "sink",
      text: "En el camino de esta salida hay un error.",
      details: errors.flatMap((error) => [error.text, error.hint]),
    });

    for (const error of errors) {
      const cause = error.causeNodeId ? canvasNodeIdOf(error.causeNodeId) : undefined;
      const where = error.nodeId ? canvasNodeIdOf(error.nodeId) : undefined;

      if (cause) {
        put(cause, { role: "cause", text: "Esta carta causó el error.", details: [error.hint] });
      }
      // Cuando coinciden, una sola marca: la del causante, que es la accionable.
      if (where && where !== cause) {
        put(where, { role: "where", text: "Aquí ocurrió el error.", details: [error.text] });
      }
    }
  }

  return marks;
}

/** Deja en cada carta su papel en el error, y se lo quita a las que ya no lo tienen. */
export function applyErrorMarks(
  nodes: DataflowNode[],
  edges: Edge[],
  errorsByOutput: Map<string, OutputErrorInfo[]>
): DataflowNode[] {
  const marks = computeErrorMarks(edges, errorsByOutput);
  let changed = false;

  const updated = nodes.map((node) => {
    const current = (node.data as { errorMark?: NodeErrorMark }).errorMark;
    const next = marks.get(node.id);

    if (safeJsonStringify(current ?? null) === safeJsonStringify(next ?? null)) return node;

    changed = true;
    // `data` es una unión discriminada por el tipo de nodo y esparcirla la
    // aplana; la marca es común a todas, así que se reafirma el tipo.
    return { ...node, data: { ...node.data, errorMark: next } } as DataflowNode;
  });

  return changed ? updated : nodes;
}
