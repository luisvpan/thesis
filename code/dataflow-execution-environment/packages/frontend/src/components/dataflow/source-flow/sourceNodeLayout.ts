import type { CardCategory } from '@/types/card-types';

/** Variantes de nodo fuente que comparten el layout actual (cubos, números, formas, etc.). */
export type SourceCardCategory =
  | Extract<CardCategory, 'number' | 'shape' | 'montessori' | 'cap' | 'stick' | 'food'>
  | 'criteria';

/** Contenedor externo del nodo en el canvas (handles + badge). Mismo valor para todos por ahora. */
const DEFAULT_WRAPPER =
  'relative h-80 w-52 -translate-x-[30%] -translate-y-[50%]';


const NUMBER_WRAPPER = 'relative h-60 w-42 -translate-x-[40%] -translate-y-[50%]';

const FOOD_WRAPPER = 'relative h-60 w-42 -translate-x-[40%] -translate-y-[50%]';

const SHAPE_WRAPPER = 'relative h-60 w-42 -translate-x-[40%] -translate-y-[50%]';

const CAP_WRAPPER = 'relative h-30 w-34 -translate-x-[35%] -translate-y-[50%]';

const MONTESORI_WRAPPER = 'relative h-30 w-34 -translate-x-[35%] -translate-y-[50%]';

const CRITERIA_WRAPPER = 'relative h-40 w-40 -translate-x-[40%] -translate-y-[50%]';

export const SOURCE_NODE_WRAPPER_CLASS: Record<SourceCardCategory, string> = {
  number: NUMBER_WRAPPER,
  shape: SHAPE_WRAPPER,
  montessori: MONTESORI_WRAPPER,
  cap: CAP_WRAPPER,
  stick: DEFAULT_WRAPPER,
  food: FOOD_WRAPPER,
  criteria: CRITERIA_WRAPPER,
};
