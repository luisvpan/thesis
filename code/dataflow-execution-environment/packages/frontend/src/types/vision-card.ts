/**
 * Tipos de cartas detectadas por el sistema de visión.
 */

import type { CapColor, FoodType, MontessoriColor, OperatorType, ShapeColor, ShapeSize, ShapeType, StickColor } from '@/types/card-types';
import { spawnActionForYoloClass } from '../data/yoloDeckCatalog';

/** Operadores matemáticos soportados */
export type VisionOperator = 'addition' | 'subtraction' | 'multiplication' | 'division';

/** Dígitos soportados (0-9) */
export type VisionDigit = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

/** Tipo de carta: número u operador */
export type VisionCardType = 'number' | 'operator';

/** Resultado del parseo de una etiqueta de visión */
export type ParsedVisionCard =
  | { type: 'number'; value: VisionDigit }
  | { type: 'operator'; operator: VisionOperator }
  | { type: 'operatorCanvas'; operator: OperatorType }
  /** Carta física detectada como `grapes`: marcador visual de salida */
  | { type: 'resultAnchor' }
  | { type: 'programResultCard' }
  | { type: 'visionArrayOpen' }
  | { type: 'visionArrayClose' }
  | { type: 'deckShape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
  | { type: 'deckFood'; yoloClass: string; food: FoodType }
  | { type: 'deckMontessori'; yoloClass: string; color: MontessoriColor }
  | { type: 'deckCap'; yoloClass: string; color: CapColor }
  | { type: 'deckStick'; yoloClass: string; color: StickColor }
  | { type: 'unknown'; label: string };

/**
 * Mapeo de etiquetas YOLO a dígitos (inglés y español)
 */
const DIGIT_LABELS: Record<string, VisionDigit> = {
  // Inglés
  zero: 0,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  // Español
  cero: 0,
  uno: 1,
  dos: 2,
  tres: 3,
  cuatro: 4,
  cinco: 5,
  seis: 6,
  siete: 7,
  ocho: 8,
  nueve: 9,
};

/**
 * Mapeo de etiquetas YOLO a operadores
 */
const OPERATOR_LABELS: Record<string, VisionOperator> = {
  // Inglés
  add: 'addition',
  addition: 'addition',
  plus: 'addition',
  subtract: 'subtraction',
  subtraction: 'subtraction',
  minus: 'subtraction',
  multiply: 'multiplication',
  multiplication: 'multiplication',
  times: 'multiplication',
  divide: 'division',
  division: 'division',
  // Español
  suma: 'addition',
  adicion: 'addition',
  resta: 'subtraction',
  sustraccion: 'subtraction',
  multiplicacion: 'multiplication',
  // 'division' ya está definido arriba (es igual en inglés y español)
};

/**
 * Parsea una etiqueta de visión YOLO y devuelve el tipo y valor de la carta.
 */
export function parseVisionLabel(label: string): ParsedVisionCard {
  const normalized = label.trim().toLowerCase();

  const spawn = spawnActionForYoloClass(normalized);
  if (spawn) {
    if (spawn.kind === 'number') return { type: 'number', value: spawn.value as VisionDigit };
    if (spawn.kind === 'resultCard') return { type: 'programResultCard' };
    if (spawn.kind === 'arrayOpen') return { type: 'visionArrayOpen' };
    if (spawn.kind === 'arrayClose') return { type: 'visionArrayClose' };
    if (spawn.kind === 'operator') {
      const op = spawn.operator;
      if (op === 'adicion') return { type: 'operator', operator: 'addition' };
      if (op === 'sustraccion') return { type: 'operator', operator: 'subtraction' };
      if (op === 'multiplicacion') return { type: 'operator', operator: 'multiplication' };
      if (op === 'division') return { type: 'operator', operator: 'division' };
      return { type: 'operatorCanvas', operator: op };
    }
    if (spawn.kind === 'shape') {
      return { type: 'deckShape', yoloClass: spawn.yoloClass, shape: spawn.shape, size: spawn.size, color: spawn.color };
    }
    if (spawn.kind === 'food') {
      // `grapes` queda como marcador físico, no como comida en el grafo.
      if (spawn.yoloClass === 'grapes') return { type: 'resultAnchor' };
      return { type: 'deckFood', yoloClass: spawn.yoloClass, food: spawn.food };
    }
    if (spawn.kind === 'montessori') {
      return { type: 'deckMontessori', yoloClass: spawn.yoloClass, color: spawn.color };
    }
    if (spawn.kind === 'cap') {
      return { type: 'deckCap', yoloClass: spawn.yoloClass, color: spawn.color };
    }
    if (spawn.kind === 'stick') {
      return { type: 'deckStick', yoloClass: spawn.yoloClass, color: spawn.color };
    }
  }

  if (normalized in DIGIT_LABELS) {
    return { type: 'number', value: DIGIT_LABELS[normalized] };
  }

  const digitMatch = /^(\d)$/.exec(normalized);
  if (digitMatch) {
    const value = Number(digitMatch[1]) as VisionDigit;
    return { type: 'number', value };
  }

  if (normalized in OPERATOR_LABELS) {
    return { type: 'operator', operator: OPERATOR_LABELS[normalized] };
  }

  return { type: 'unknown', label };
}

/**
 * Convierte VisionOperator al tipo de operador usado internamente (MathOperatorType).
 */
export function visionOperatorToMathOperator(op: VisionOperator): 'adicion' | 'sustraccion' | 'multiplicacion' | 'division' {
  const mapping: Record<VisionOperator, 'adicion' | 'sustraccion' | 'multiplicacion' | 'division'> = {
    addition: 'adicion',
    subtraction: 'sustraccion',
    multiplication: 'multiplicacion',
    division: 'division',
  };
  return mapping[op];
}
