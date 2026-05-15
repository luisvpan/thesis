import type { FoodType, MontessoriColor, OperatorType, ShapeColor, ShapeSize, ShapeType } from '@/types/card-types';

export type DeckSectionId = 'numbers' | 'operators' | 'figures' | 'foods' | 'montessori' | 'arrayMarkers';

export const DECK_SECTION_ITEMS: Record<DeckSectionId, readonly string[]> = {
  numbers: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
  operators: ['add', 'subtract', 'multiply', 'division', 'ascending', 'descending', 'filter', 'output'],
  figures: [
    'sm_circle',
    'sm_square',
    'sm_triangle',
    'md_circle',
    'md_square',
    'md_triangle',
    'lg_circle',
    'lg_square',
    'lg_triangle',
    'small',
    'medium',
    'large',
    'green',
    'purple',
    'red',
  ],
  foods: ['apple', 'burger', 'pear', 'grapes', 'orange'],
  montessori: ['montessori_blue', 'montessori_green', 'montessori_orange', 'montessori_purple', 'montessori_red', 'montessori_yellow'],
  arrayMarkers: ['open', 'close'],
};

/** Operadores permitidos en concreto y pictórico (suma, resta, filtrar, ordenar). */
const YOLO_OPERATORS_CPA_CONCRETE_PICTORIAL = ['add', 'subtract', 'filter', 'ascending', 'descending'] as const;

const YOLO_OPERATORS_ABSTRACT = [
  'add',
  'subtract',
  'multiply',
  'division',
  'ascending',
  'descending',
] as const;

export type CpaSidebarSectionId = 'concreto' | 'pictorico' | 'abstracto' | 'comun';

/** Cartas agrupadas por CPA (mismo criterio que el lenguaje) + sección común (salida y arreglos). */
export const CPA_YOLO_SIDEBAR_SECTIONS: ReadonlyArray<{
  id: CpaSidebarSectionId;
  title: string;
  yoloClasses: readonly string[];
}> = [
  {
    id: 'concreto',
    title: 'Concreto',
    yoloClasses: [
      ...DECK_SECTION_ITEMS.foods,
      ...DECK_SECTION_ITEMS.montessori,
      ...YOLO_OPERATORS_CPA_CONCRETE_PICTORIAL,
    ],
  },
  {
    id: 'pictorico',
    title: 'Pictórico',
    yoloClasses: [...DECK_SECTION_ITEMS.figures, ...YOLO_OPERATORS_CPA_CONCRETE_PICTORIAL],
  },
  {
    id: 'abstracto',
    title: 'Abstracto',
    yoloClasses: [...DECK_SECTION_ITEMS.numbers, ...YOLO_OPERATORS_ABSTRACT],
  },
  {
    id: 'comun',
    title: 'Común',
    yoloClasses: ['output', ...DECK_SECTION_ITEMS.arrayMarkers],
  },
];

export function deckLabel(yoloClass: string): string {
  const m: Record<string, string> = {
    zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9',
    add: '+', subtract: '-', multiply: 'x', division: '÷', ascending: 'Asc', descending: 'Desc', filter: 'Filtrar',
    output: 'Resultado', result: 'Resultado',
    open: 'Abrir', close: 'Cerrar',
    small: 'pequeño', medium: 'mediano', large: 'grande',
    sm_circle: 'Circulo P', sm_square: 'Cuadrado P', sm_triangle: 'Triángulo P',
    md_circle: 'Circulo M', md_square: 'Cuadrado M', md_triangle: 'Triángulo M',
    lg_circle: 'Circulo G', lg_square: 'Cuadrado G', lg_triangle: 'Triángulo G',
    green: 'Verde', purple: 'Morado', red: 'Rojo',
    apple: 'Manzana', burger: 'Hamburguesa', pear: 'Pera', grapes: 'Uvas', orange: 'Naranja',
    montessori_blue: 'Montessori Azul', montessori_green: 'Montessori Verde', montessori_orange: 'Montessori Naranja',
    montessori_purple: 'Montessori Morado', montessori_red: 'Montessori Rojo', montessori_yellow: 'Montessori Amarillo',
  };
  return m[yoloClass] ?? yoloClass;
}

export type DeckSpawnAction =
  | { kind: 'number'; value: number }
  | { kind: 'operator'; operator: OperatorType }
  | { kind: 'resultCard' }
  | { kind: 'arrayOpen' }
  | { kind: 'arrayClose' }
  | { kind: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
  | { kind: 'food'; yoloClass: string; food: FoodType }
  | { kind: 'montessori'; yoloClass: string; color: MontessoriColor };

export function spawnActionForYoloClass(raw: string): DeckSpawnAction | null {
  const x = raw.trim().toLowerCase();
  const digit: Record<string, number> = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
  if (x in digit) return { kind: 'number', value: digit[x] };
  if (x === 'output' || x === 'result') return { kind: 'resultCard' };
  if (x === 'open') return { kind: 'arrayOpen' };
  if (x === 'close') return { kind: 'arrayClose' };

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
    sm_circle: { shape: 'circulo', size: 'pequeño', color: 'amarillo' },
    md_circle: { shape: 'circulo', size: 'mediano', color: 'amarillo' },
    lg_circle: { shape: 'circulo', size: 'grande', color: 'amarillo' },
    sm_square: { shape: 'cuadrado', size: 'pequeño', color: 'amarillo' },
    md_square: { shape: 'cuadrado', size: 'mediano', color: 'amarillo' },
    lg_square: { shape: 'cuadrado', size: 'grande', color: 'amarillo' },
    sm_triangle: { shape: 'triangulo', size: 'pequeño', color: 'amarillo' },
    md_triangle: { shape: 'triangulo', size: 'mediano', color: 'amarillo' },
    lg_triangle: { shape: 'triangulo', size: 'grande', color: 'amarillo' },
    small: { shape: 'circulo', size: 'pequeño', color: 'amarillo' },
    medium: { shape: 'circulo', size: 'mediano', color: 'amarillo' },
    large: { shape: 'circulo', size: 'grande', color: 'amarillo' },
    green: { shape: 'circulo', size: 'mediano', color: 'verde' },
    purple: { shape: 'circulo', size: 'mediano', color: 'azul' },
    red: { shape: 'circulo', size: 'mediano', color: 'rojo' },
  };
  if (x in fig) return { kind: 'shape', yoloClass: x, ...fig[x] };

  const mont: Record<string, MontessoriColor> = {
    montessori_blue: 'azul',
    montessori_green: 'verde',
    montessori_orange: 'naranja',
    montessori_purple: 'morado',
    montessori_red: 'rojo',
    montessori_yellow: 'amarillo',
  };
  if (x in mont) return { kind: 'montessori', yoloClass: x, color: mont[x] };

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
