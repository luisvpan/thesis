// Tipos para el canvas de piezas y conexiones (Sandbox / Modo Jugar)

export interface Piece {
  id: string;
  type: 'number' | 'operation';
  value: string | number;
  x: number;
  y: number;
}

export interface Connection {
  id: string;
  from: string;
  to: string;
  active: boolean;
}
