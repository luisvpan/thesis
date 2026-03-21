import type { Node } from '@xyflow/react';

/**
 * Eventos que el frontend escucha (enviados por el backend):
 *
 * - addNode  → agrega un nodo al canvas (payload: SocketAddNodePayload)
 * - addEdge  → agrega una conexión entre nodos (payload: SocketAddEdgePayload)
 * - navigate → navega sin mouse (payload: SocketNavigatePayload)
 */

/** Payload que envía el backend para agregar un nodo al canvas */
export interface SocketAddNodePayload {
  id: string;
  type: 'number' | 'operator';
  position: { x: number; y: number };
  data: Record<string, unknown>;
}

/** Payload que envía el backend para agregar una conexión (edge) */
export interface SocketAddEdgePayload {
  id?: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

/** Nodo en formato que acepta React Flow (compatible con Node<>). */
export type SocketNode = Node<Record<string, unknown>, 'number' | 'operator'>;

/** Nombres de botones de navegación que el backend puede enviar */
export type NavigateTo =
  | 'menu'
  | 'configuracion'
  | 'atras'
  | 'jugar'
  | 'mundos'
  | 'niveles';

/** Payload del evento 'navigate': path directo o acción nombrada (sin mouse). */
export interface SocketNavigatePayload {
  /** Ruta directa, ej: "/", "/juego", "/juego/1", "/ide/sandbox" */
  path?: string;
  /** Acción nombrada (se traduce a path en el frontend) */
  to?: NavigateTo;
  /** Requerido si to === 'niveles', ej: "1" para Mundo 1 */
  worldId?: string;
}
