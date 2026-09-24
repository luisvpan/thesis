// Firmas de las operaciones — LANGUAGE_SPEC.md §3 y §5.2
//
// Tabla única: de aquí salen tanto el despacho en ejecución como los chequeos
// de aridad, categoría y subtipo de criterio de la pasada estática (§4.2.4-7).

import type { CriterionSubtype, ValueCategory } from "../runtime/types";

/** Las operaciones reconocidas (§5.2). Cualquier otro nombre es un error estático. */
export const OPERATIONS = [
  "sum",
  "substract",
  "multiply",
  "divide",
  "less_than",
  "greater_than",
  "compare",
  "order",
  "filter",
  "first",
  "last",
  "count",
] as const;

export type Operation = (typeof OPERATIONS)[number];

export function isOperation(name: string): name is Operation {
  return (OPERATIONS as readonly string[]).includes(name);
}

export interface ParameterSpec {
  category: ValueCategory;
  /** Solo para criterios: el subtipo que admite esta posición (§4.2.7). */
  criterion?: CriterionSubtype;
  /** La bolsa debe resultar ser un número al evaluarse (§4.3.1). */
  number?: boolean;
}

export interface OperationSignature {
  /** Parámetros fijos, en orden. */
  parameters: ParameterSpec[];
  /** Forma de los argumentos que siguen a los fijos, si la operación es variádica. */
  rest?: ParameterSpec;
  minArity: number;
  /** `null` = sin tope. */
  maxArity: number | null;
  result: ValueCategory;
  /**
   * Si agrupa por identidad antes de actuar (§3, convenciones). Todas menos la
   * aritmética que suma (`sum`, `substract`) dejan fuera las entradas
   * abstractas: cada carta de número se escala, se ordena o se compara por
   * separado.
   */
  aggregates: boolean;
}

const BOLSA: ParameterSpec = { category: "bolsa" };
const NUMERO: ParameterSpec = { category: "bolsa", number: true };
const CRITERIO_FILTRO: ParameterSpec = { category: "criterio", criterion: "filter" };
const CRITERIO_ORDEN: ParameterSpec = { category: "criterio", criterion: "order" };

export const SIGNATURES: Record<Operation, OperationSignature> = {
  // §3.1 Aritmética
  sum: { parameters: [], rest: BOLSA, minArity: 1, maxArity: null, result: "bolsa", aggregates: true },
  substract: { parameters: [BOLSA, BOLSA], minArity: 2, maxArity: 2, result: "bolsa", aggregates: true },
  multiply: { parameters: [BOLSA, NUMERO], minArity: 2, maxArity: 2, result: "bolsa", aggregates: true },
  divide: { parameters: [BOLSA, NUMERO], minArity: 2, maxArity: 2, result: "bolsa", aggregates: true },

  // §3.2 Comparación
  less_than: { parameters: [BOLSA, NUMERO], minArity: 2, maxArity: 2, result: "bolsa", aggregates: true },
  greater_than: { parameters: [BOLSA, NUMERO], minArity: 2, maxArity: 2, result: "bolsa", aggregates: true },
  compare: { parameters: [BOLSA, BOLSA], minArity: 2, maxArity: 2, result: "booleano", aggregates: false },

  // §3.3 Orden
  order: {
    parameters: [BOLSA],
    rest: CRITERIO_ORDEN,
    minArity: 2,
    maxArity: null,
    result: "bolsa",
    aggregates: true,
  },

  // §3.4 Filtrado
  filter: {
    parameters: [BOLSA],
    rest: CRITERIO_FILTRO,
    minArity: 2,
    maxArity: null,
    result: "bolsa",
    aggregates: false,
  },

  // §3.5 Acceso — entrada por entrada: señalan una pila individual
  first: { parameters: [BOLSA], minArity: 1, maxArity: 1, result: "bolsa", aggregates: false },
  last: { parameters: [BOLSA], minArity: 1, maxArity: 1, result: "bolsa", aggregates: false },

  // §3.6 Agregación
  count: { parameters: [BOLSA], minArity: 1, maxArity: 1, result: "bolsa", aggregates: false },
};

/** La forma que debe tener el argumento en la posición `index`. */
export function parameterAt(signature: OperationSignature, index: number): ParameterSpec | undefined {
  return signature.parameters[index] ?? signature.rest;
}

/** La aridad admitida, en español, para el mensaje del error (§4.2.5). */
export function describeArity(signature: OperationSignature): string {
  const { minArity, maxArity } = signature;
  if (maxArity === null) return `al menos ${minArity}`;
  if (minArity === maxArity) return `exactamente ${minArity}`;
  return `entre ${minArity} y ${maxArity}`;
}
