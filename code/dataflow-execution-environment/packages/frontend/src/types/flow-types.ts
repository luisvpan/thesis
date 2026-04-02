// Tipos para el sistema de flujo de datos (dataflow)

import { Card } from './card-types';

export type VisualizationMode = 'real' | 'pictorico' | 'abstracto';

export type FlowNodeType = 'card' | 'result';

export interface Position {
  x: number;
  y: number;
}

// Resultado de una operación
export interface FlowResult {
  value: any; // El valor calculado
  type: 'number' | 'array' | 'shape' | 'animal' | 'car' | 'person';
  count?: number; // Cantidad de elementos si es un array
  items?: any[]; // Items individuales si es un array
}

// Nodo de carta (entrada)
export interface CardNode {
  id: string;
  nodeType: 'card';
  card: Card;
  position: Position;
}

// Nodo de resultado (salida de operación)
export interface ResultNode {
  id: string;
  nodeType: 'result';
  result: FlowResult | null;
  position: Position;
  inputConnections: string[]; // IDs de las conexiones que entran a este nodo
}

export type FlowNode = CardNode | ResultNode;

// Conexión entre nodos
export interface FlowConnection {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  active: boolean;
  resultNodeId?: string; // ID del nodo de resultado que genera esta conexión
}

// Estado global del flujo
export interface FlowState {
  nodes: FlowNode[];
  connections: FlowConnection[];
  visualizationMode: VisualizationMode;
}
