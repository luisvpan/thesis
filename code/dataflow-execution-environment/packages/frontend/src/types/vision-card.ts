/**
 * Tipos de cartas detectadas por el sistema de visión.
 */

import type { OperatorType } from '@/types/card-types';
import type { ShapeType, ShapeSize, ShapeColor } from '@/types/card-types';
import type { FoodType } from '@/types/card-types';
import { spawnActionForYoloClass } from '@/data/yoloDeckCatalog';

/** Operadores matemáticos soportados en el DSL / resultado numérico */
export type VisionOperator = 'addition' | 'subtraction' | 'multiplication' | 'division';

/** Dígitos soportados (0-9) */
export type VisionDigit = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

/** Tipo de carta: número u operador */
export type VisionCardType = 'number' | 'operator';

/** Resultado del parseo de una etiqueta de visión */
export type ParsedVisionCard =
  | { type: 'number'; value: VisionDigit }
  | { type: 'operator'; operator: VisionOperator }
  /** Carta física `grapes`: marcador + carta resultado (uva). */
  | { type: 'resultAnchor' }
  /** Carta clase `result`: una sola carta resultado (como modo dev). */
  | { type: 'programResultCard' }
  /** Operador no solo matemático (orden, filtro, …). */
  | { type: 'operatorCanvas'; operator: OperatorType }
  | {
      type: 'deckShape';
      yoloClass: string;
      shape: ShapeType;
      size: ShapeSize;
      color: ShapeColor;
    }
  | { type: 'deckFood'; yoloClass: string; food: FoodType }
  | { type: 'unknown'; label: string };

const DIGIT_LABELS: Record<string, VisionDigit> = {
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

const OPERATOR_LABELS: Record<string, VisionOperator> = {
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
  suma: 'addition',
  adicion: 'addition',
  resta: 'subtraction',
  sustraccion: 'subtraction',
  multiplicacion: 'multiplication',
};

function operatorTypeToVision(op: OperatorType): VisionOperator | null {
  const m: Partial<Record<OperatorType, VisionOperator>> = {
    adicion: 'addition',
    sustraccion: 'subtraction',
    multiplicacion: 'multiplication',
    division: 'division',
  };
  return m[op] ?? null;
}

/**
 * Parsea una etiqueta YOLO / visión y devuelve el tipo de carta para el lienzo.
 */
export function parseVisionLabel(label: string): ParsedVisionCard {
  const normalized = label.trim().toLowerCase();

  if (normalized === 'grapes' || normalized === 'grape') {
    return { type: 'resultAnchor' };
  }

  const spawn = spawnActionForYoloClass(normalized);
  if (spawn) {
    if (spawn.kind === 'number') {
      const v = Math.max(0, Math.min(9, spawn.value)) as VisionDigit;
      return { type: 'number', value: v };
    }
    if (spawn.kind === 'resultCard') {
      return { type: 'programResultCard' };
    }
    if (spawn.kind === 'operator') {
      const vo = operatorTypeToVision(spawn.operator);
      if (vo) {
        return { type: 'operator', operator: vo };
      }
      return { type: 'operatorCanvas', operator: spawn.operator };
    }
    if (spawn.kind === 'shape') {
      return {
        type: 'deckShape',
        yoloClass: spawn.yoloClass,
        shape: spawn.shape,
        size: spawn.size,
        color: spawn.color,
      };
    }
    if (spawn.kind === 'food') {
      return {
        type: 'deckFood',
        yoloClass: spawn.yoloClass,
        food: spawn.food,
      };
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
export function visionOperatorToMathOperator(
  op: VisionOperator
): 'adicion' | 'sustraccion' | 'multiplicacion' | 'division' {
  const mapping: Record<
    VisionOperator,
    'adicion' | 'sustraccion' | 'multiplicacion' | 'division'
  > = {
    addition: 'adicion',
    subtraction: 'sustraccion',
    multiplication: 'multiplicacion',
    division: 'division',
  };
  return mapping[op];
}
