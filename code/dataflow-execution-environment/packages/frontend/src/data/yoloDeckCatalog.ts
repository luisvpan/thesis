import type { CapColor, FoodType, MontessoriColor, OperatorType, ShapeColor, ShapeSize, ShapeType, StickColor } from '@/types/card-types';

export type DeckSectionId = 'numbers' | 'operators' | 'figures' | 'foods' | 'montessori' | 'caps' | 'sticks' | 'arrayMarkers' | 'dice';

export const DECK_SECTION_ITEMS: Record<DeckSectionId, readonly string[]> = {
  numbers: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
  operators: ['add', 'subtract', 'multiply', 'division', 'ascending', 'descending', 'smallest_to_largest', 'largest_to_smallest', 'filter', 'compare', 'first', 'last', 'count', 'sink'],
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
    // Criteria cards (size/color/subtype)
    'small',
    'medium',
    'large',
    'green',
    'purple',
    'red',
    'orange',
    'blue',
    'yellow',
    'circle',
    'square',
    'triangle',
  ],
  foods: ['apple', 'burger', 'pear', 'grapes'],
  montessori: ['cube_blue', 'cube_red', 'cube_yellow'],
  caps: ['cap_blue', 'cap_white'],
  sticks: ['stick_cyan', 'stick_orange', 'stick_red', 'stick_wooden'],
  arrayMarkers: ['open', 'close'],
  dice: ['dice'],
};

/** Operadores permitidos en concreto y pictórico (suma, resta, filtrar, ordenar). */
const YOLO_OPERATORS_CPA_CONCRETE_PICTORIAL = ['add', 'subtract', 'filter', 'ascending', 'descending', 'compare', 'first', 'last', 'count'] as const;

const YOLO_OPERATORS_ABSTRACT = [
  'add',
  'subtract',
  'multiply',
  'division',
  'ascending',
  'descending',
  'compare',
  'first',
  'last',
  'count',
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
      ...DECK_SECTION_ITEMS.caps,
      ...DECK_SECTION_ITEMS.sticks,
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
    yoloClasses: ['sink', ...DECK_SECTION_ITEMS.arrayMarkers, ...DECK_SECTION_ITEMS.dice],
  },
];

export function deckLabel(yoloClass: string): string {
  const m: Record<string, string> = {
    zero: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9',
    add: '+', subtract: '-', multiply: 'x', division: '÷',
    ascending: 'Asc', descending: 'Desc',
    smallest_to_largest: '↑ Tamaño', largest_to_smallest: '↓ Tamaño',
    filter: 'Filtrar', compare: '=?', first: 'Primero', last: 'Último', count: 'Contar',
    sink: 'Resultado', output: 'Resultado', result: 'Resultado',
    open: 'Abrir', close: 'Cerrar',
    dice: 'Dado',
    // Criteria de tamaño
    small: 'Pequeño', medium: 'Mediano', large: 'Grande',
    // Criteria de color
    green: 'Verde', purple: 'Morado', red: 'Rojo', orange: 'Naranja', blue: 'Azul', yellow: 'Amarillo',
    // Criteria de forma
    circle: 'Círculo', square: 'Cuadrado', triangle: 'Triángulo',
    // Figuras
    sm_circle: 'Circulo P', sm_square: 'Cuadrado P', sm_triangle: 'Triángulo P',
    md_circle: 'Circulo M', md_square: 'Cuadrado M', md_triangle: 'Triángulo M',
    lg_circle: 'Circulo G', lg_square: 'Cuadrado G', lg_triangle: 'Triángulo G',
    // Alimentos
    apple: 'Manzana', burger: 'Hamburguesa', pear: 'Pera', grapes: 'Uvas',
    // Cubos Montessori
    cube_blue: 'Cubo Azul', cube_red: 'Cubo Rojo', cube_yellow: 'Cubo Amarillo',
    // Tapas
    cap_blue: 'Tapa Azul', cap_white: 'Tapa Blanca',
    // Paletas
    stick_cyan: 'Paleta Cian',
    stick_orange: 'Paleta Naranja',
    stick_red: 'Paleta Roja',
    stick_wooden: 'Paleta de Madera',
  };
  return m[yoloClass] ?? yoloClass;
}

export type CriteriaProperty = 'size' | 'color' | 'subtype';

export type CriteriaValues = {
  size?: 'pequeño' | 'mediano' | 'grande';
  color?: 'verde' | 'morado' | 'rojo' | 'naranja' | 'azul' | 'amarillo';
  subtype?: 'circulo' | 'cuadrado' | 'triangulo';
};

export type OrderCriterio = {
  property: CriteriaProperty;
  sequence: string[];
};

export type DeckSpawnAction =
  | { kind: 'number'; value: number }
  | { kind: 'operator'; operator: OperatorType; criterio?: OrderCriterio }
  | { kind: 'resultCard' }
  | { kind: 'arrayOpen' }
  | { kind: 'arrayClose' }
  | { kind: 'dice' }
  | { kind: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
  | { kind: 'food'; yoloClass: string; food: FoodType }
  | { kind: 'montessori'; yoloClass: string; color: MontessoriColor }
  | { kind: 'cap'; yoloClass: string; color: CapColor }
  | { kind: 'stick'; yoloClass: string; color?: StickColor }
  | { kind: 'criteria'; yoloClass: string; properties: CriteriaProperty[]; values: CriteriaValues };

export function spawnActionForYoloClass(raw: string): DeckSpawnAction | null {
  const x = raw.trim().toLowerCase();
  const digit: Record<string, number> = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
  if (x in digit) return { kind: 'number', value: digit[x] };
  if (x === 'sink' || x === 'output' || x === 'result') return { kind: 'resultCard' };
  if (x === 'open') return { kind: 'arrayOpen' };
  if (x === 'close') return { kind: 'arrayClose' };
  if (x === 'dice') return { kind: 'dice' };

  // Operadores básicos
  const op: Record<string, OperatorType> = {
    add: 'adicion',
    subtract: 'sustraccion',
    multiply: 'multiplicacion',
    division: 'division',
    ascending: 'orden-menor-mayor',
    descending: 'orden-mayor-menor',
    filter: 'filtrar-general',
    compare: 'comparar',
    first: 'primero',
    last: 'ultimo',
    count: 'contar',
  };
  if (x in op) return { kind: 'operator', operator: op[x] };

  // Operadores de ordenamiento por size (con criterio implícito)
  const SIZE_SEQUENCE = ['pequeño', 'mediano', 'grande'];
  if (x === 'smallest_to_largest') {
    return {
      kind: 'operator',
      operator: 'orden-menor-mayor',
      criterio: { property: 'size', sequence: SIZE_SEQUENCE },
    };
  }
  if (x === 'largest_to_smallest') {
    return {
      kind: 'operator',
      operator: 'orden-mayor-menor',
      criterio: { property: 'size', sequence: SIZE_SEQUENCE },
    };
  }

  // Colores pictóricos → Criteria literals (no shapes)
  const colorCriteria: Record<string, CriteriaValues['color']> = {
    green: 'verde',
    purple: 'morado',
    red: 'rojo',
    orange: 'naranja',
    blue: 'azul',
    yellow: 'amarillo',
  };
  if (x in colorCriteria) {
    return {
      kind: 'criteria',
      yoloClass: x,
      properties: ['color'],
      values: { color: colorCriteria[x] },
    };
  }

  // Tamaños → Criteria literals (no shapes)
  const sizeCriteria: Record<string, CriteriaValues['size']> = {
    small: 'pequeño',
    medium: 'mediano',
    large: 'grande',
  };
  if (x in sizeCriteria) {
    return {
      kind: 'criteria',
      yoloClass: x,
      properties: ['size'],
      values: { size: sizeCriteria[x] },
    };
  }

  const subtypeCriteria: Record<string, CriteriaValues['subtype']> = {
    circle: 'circulo',
    square: 'cuadrado',
    triangle: 'triangulo',
    circulo: 'circulo',
    cuadrado: 'cuadrado',
    triangulo: 'triangulo',
  };
  if (x in subtypeCriteria) {
    return {
      kind: 'criteria',
      yoloClass: x,
      properties: ['subtype'],
      values: { subtype: subtypeCriteria[x] },
    };
  }

  // Figuras con tamaño específico (estas sí son shapes)
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
  };
  if (x in fig) return { kind: 'shape', yoloClass: x, ...fig[x] };

  // Cubos Montessori (cube_blue, cube_red, cube_yellow)
  const mont: Record<string, MontessoriColor> = {
    cube_blue: 'azul',
    cube_red: 'rojo',
    cube_yellow: 'amarillo',
  };
  if (x in mont) return { kind: 'montessori', yoloClass: x, color: mont[x] };

  // Tapas (cap_blue, cap_white)
  const caps: Record<string, CapColor> = {
    cap_blue: 'azul',
    cap_white: 'blanco',
  };
  if (x in caps) return { kind: 'cap', yoloClass: x, color: caps[x] };

  // Paletas de color (stick_cyan, stick_orange, stick_red)
  const sticks: Record<string, StickColor> = {
    stick_cyan: 'cian',
    stick_orange: 'naranja',
    stick_red: 'rojo',
  };
  if (x in sticks) return { kind: 'stick', yoloClass: x, color: sticks[x] };

  // Paleta de madera (stick_wooden)
  if (x === 'stick_wooden') return { kind: 'stick', yoloClass: x, color: 'madera' };

  const food: Record<string, FoodType> = {
    apple: 'manzana',
    burger: 'hamburguesa',
    pear: 'peras',
    grapes: 'uvas',
  };
  if (x in food) return { kind: 'food', yoloClass: x, food: food[x] };

  return null;
}
