import type { FoodType, OperatorType, ShapeColor, ShapeSize, ShapeType } from '@/types/card-types';

export type DeckSectionId = 'numbers' | 'operators' | 'figures' | 'foods';

export const DECK_SECTION_ITEMS: Record<DeckSectionId, readonly string[]> = {
  numbers: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
  operators: ['add', 'subtract', 'multiply', 'division', 'ascending', 'descending', 'filter', 'result'],
  figures: ['sm_circle', 'sm_square', 'md_circle', 'md_square', 'lg_circle', 'lg_square', 'green', 'purple', 'red'],
  foods: ['apple', 'burger', 'pear', 'grapes', 'orange'],
};

export function deckLabel(yoloClass: string): string {
  const m: Record<string, string> = {
    zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9',
    add: '+', subtract: '-', multiply: 'x', division: '÷', ascending: 'Asc', descending: 'Desc', filter: 'Filtrar', result: 'Resultado',
    sm_circle: 'Circulo P', sm_square: 'Cuadrado P', md_circle: 'Circulo M', md_square: 'Cuadrado M', lg_circle: 'Circulo G', lg_square: 'Cuadrado G',
    green: 'Verde', purple: 'Morado', red: 'Rojo',
    apple: 'Manzana', burger: 'Hamburguesa', pear: 'Pera', grapes: 'Uvas', orange: 'Naranja',
  };
  return m[yoloClass] ?? yoloClass;
}

export type DeckSpawnAction =
  | { kind: 'number'; value: number }
  | { kind: 'operator'; operator: OperatorType }
  | { kind: 'resultCard' }
  | { kind: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
  | { kind: 'food'; yoloClass: string; food: FoodType };

export function spawnActionForYoloClass(raw: string): DeckSpawnAction | null {
  const x = raw.trim().toLowerCase();
  const digit: Record<string, number> = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
  if (x in digit) return { kind: 'number', value: digit[x] };
  if (x === 'result') return { kind: 'resultCard' };

  const op: Record<string, OperatorType> = {
    add: 'adicion',
    subtract: 'sustraccion',
    multiply: 'multiplicacion',
    division: 'division',
    ascending: 'orden-menor-mayor',
    descending: 'orden-mayor-menor',
    filter: 'filtrar-general',
  };
  if (x in op) return { kind: 'operator', operator: op[x] };

  const fig: Record<string, { shape: ShapeType; size: ShapeSize; color: ShapeColor }> = {
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
  if (x in fig) return { kind: 'shape', yoloClass: x, ...fig[x] };

  const food: Record<string, FoodType> = {
    apple: 'manzana',
    burger: 'hamburguesa',
    pear: 'peras',
    grapes: 'uvas',
    orange: 'naranja',
  };
  if (x in food) return { kind: 'food', yoloClass: x, food: food[x] };

  return null;
}
