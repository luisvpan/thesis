/**
 * Etiquetas del modelo YOLO (nc=32) agrupadas para la mochila del IDE y el parseo de visión.
 */

import type { OperatorType } from '@/types/card-types';
import type { ShapeType, ShapeSize, ShapeColor } from '@/types/card-types';
import type { FoodType } from '@/types/card-types';

export type DeckSectionId = 'numbers' | 'operators' | 'figures' | 'foods';

/** Clases numéricas (nombre YOLO → dígito). */
export const YOLO_DIGIT_CLASSES = [
  'zero',
  'one',
  'two',
  'three',
  'four',
  'five',
  'six',
  'seven',
  'eight',
  'nine',
] as const;

export type YoloDigitClass = (typeof YOLO_DIGIT_CLASSES)[number];

const DIGIT_CLASS_TO_VALUE: Record<YoloDigitClass, number> = {
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
};

/** Operadores rojos (+ orden / filtro / carta resultado). Etiquetas = clases YOLO. */
export const YOLO_OPERATOR_CLASSES = [
  'add',
  'subtract',
  'multiply',
  'division',
  'ascending',
  'descending',
  'filter',
  'result',
] as const;

/** Figuras amarillas (formas por tamaño + fichas de color del modelo). */
export const YOLO_FIGURE_CLASSES = [
  'sm_circle',
  'sm_square',
  'md_circle',
  'md_square',
  'lg_circle',
  'lg_square',
  'green',
  'purple',
  'red',
] as const;

/** Comidas naranjas (uva = comida en mesa; en visión la clase `grapes` sigue siendo marcador uva). */
export const YOLO_FOOD_CLASSES = ['apple', 'burger', 'pear', 'grapes', 'orange'] as const;

export type YoloDeckClass =
  | YoloDigitClass
  | (typeof YOLO_OPERATOR_CLASSES)[number]
  | (typeof YOLO_FIGURE_CLASSES)[number]
  | (typeof YOLO_FOOD_CLASSES)[number];

export type DeckSpawnAction =
  | { kind: 'number'; value: number }
  | { kind: 'operator'; operator: OperatorType }
  | { kind: 'resultCard' }
  | {
      kind: 'shape';
      yoloClass: string;
      shape: ShapeType;
      size: ShapeSize;
      color: ShapeColor;
    }
  | { kind: 'food'; yoloClass: string; food: FoodType };

/** Mapa YOLO → tipo interno OperatorType (subset usado en el lienzo). */
export const YOLO_TO_OPERATOR_TYPE: Record<string, OperatorType> = {
  add: 'adicion',
  subtract: 'sustraccion',
  multiply: 'multiplicacion',
  division: 'division',
  ascending: 'orden-menor-mayor',
  descending: 'orden-mayor-menor',
  filter: 'filtrar-general',
};

/** Figuras: tamaño desde prefijo sm/md/lg; color para clases green/purple/red. */
const FIGURE_SHAPE_MAP: Record<
  string,
  { shape: ShapeType; size: ShapeSize; color: ShapeColor }
> = {
  sm_circle: { shape: 'circulo', size: 'pequeña', color: 'amarillo' },
  md_circle: { shape: 'circulo', size: 'mediana', color: 'amarillo' },
  lg_circle: { shape: 'circulo', size: 'grande', color: 'amarillo' },
  sm_square: { shape: 'cuadrado', size: 'pequeña', color: 'amarillo' },
  md_square: { shape: 'cuadrado', size: 'mediana', color: 'amarillo' },
  lg_square: { shape: 'cuadrado', size: 'grande', color: 'amarillo' },
  green: { shape: 'circulo', size: 'mediana', color: 'verde' },
  purple: { shape: 'circulo', size: 'mediana', color: 'azul' },
  red: { shape: 'circulo', size: 'mediana', color: 'rojo' },
};

const FOOD_MAP: Record<string, FoodType> = {
  apple: 'manzana',
  burger: 'hamburguesa',
  pear: 'peras',
  grapes: 'uvas',
  orange: 'naranja',
};

const SPANISH_LABELS: Record<string, string> = {
  zero: '0',
  one: '1',
  two: '2',
  three: '3',
  four: '4',
  five: '5',
  six: '6',
  seven: '7',
  eight: '8',
  nine: '9',
  add: '+',
  subtract: '−',
  multiply: '×',
  division: '÷',
  ascending: 'Ascendente',
  descending: 'Descendente',
  filter: 'Filtrar',
  result: 'Resultado',
  sm_circle: 'Círculo P',
  sm_square: 'Cuadrado P',
  md_circle: 'Círculo M',
  md_square: 'Cuadrado M',
  lg_circle: 'Círculo G',
  lg_square: 'Cuadrado G',
  green: 'Verde',
  purple: 'Morado',
  red: 'Rojo',
  apple: 'Manzana',
  burger: 'Hamburguesa',
  pear: 'Pera',
  grapes: 'Uvas',
  orange: 'Naranja',
};

export function deckLabel(yoloClass: string): string {
  return SPANISH_LABELS[yoloClass] ?? yoloClass;
}

export function spawnActionForYoloClass(normalized: string): DeckSpawnAction | null {
  const n = normalized.trim().toLowerCase();
  if ((YOLO_DIGIT_CLASSES as readonly string[]).includes(n)) {
    return {
      kind: 'number',
      value: DIGIT_CLASS_TO_VALUE[n as YoloDigitClass],
    };
  }
  if (n === 'result') {
    return { kind: 'resultCard' };
  }
  const op = YOLO_TO_OPERATOR_TYPE[n];
  if (op) {
    return { kind: 'operator', operator: op };
  }
  if (FIGURE_SHAPE_MAP[n]) {
    const fig = FIGURE_SHAPE_MAP[n];
    return {
      kind: 'shape',
      yoloClass: n,
      shape: fig.shape,
      size: fig.size,
      color: fig.color,
    };
  }
  const food = FOOD_MAP[n];
  if (food) {
    return { kind: 'food', yoloClass: n, food };
  }
  return null;
}

export function sectionForYoloClass(n: string): DeckSectionId | null {
  const x = n.trim().toLowerCase();
  if ((YOLO_DIGIT_CLASSES as readonly string[]).includes(x)) return 'numbers';
  if ((YOLO_OPERATOR_CLASSES as readonly string[]).includes(x as never)) return 'operators';
  if ((YOLO_FIGURE_CLASSES as readonly string[]).includes(x as never)) return 'figures';
  if ((YOLO_FOOD_CLASSES as readonly string[]).includes(x as never)) return 'foods';
  return null;
}

/** Listas ordenadas para la mochila (una entrada por clase YOLO). */
export const DECK_SECTION_ITEMS: Record<DeckSectionId, readonly string[]> = {
  numbers: [...YOLO_DIGIT_CLASSES],
  operators: [...YOLO_OPERATOR_CLASSES],
  figures: [...YOLO_FIGURE_CLASSES],
  foods: [...YOLO_FOOD_CLASSES],
};
