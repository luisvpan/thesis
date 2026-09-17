// Construcción de valores: un único factory para bolsas, más uno por subtipo de
// criterio. La bolsa que devuelve `createBag()` es **inmutable**: cada `.add()`
// produce una bolsa nueva, y `.reset()` devuelve la vacía (`nulo`).

import Fraction from "fraction.js";
import type {
  BagLiteral,
  CriterionLiteral,
} from "./program";
import type { CPACategory, CriterionValue, Entry } from "./runtime/types";

/** Lo que hace falta para declarar una entrada: su identidad y su cantidad. */
export interface EntrySpec {
  category: CPACategory;
  type: string;
  subtype: string;
  quantity: number | string | Fraction;
  attributes?: Record<string, string>;
}

function toEntry(spec: EntrySpec): Entry {
  return Object.freeze({
    category: spec.category,
    type: spec.type,
    subtype: spec.subtype,
    attributes: Object.freeze({ ...(spec.attributes ?? {}) }),
    quantity: spec.quantity instanceof Fraction ? spec.quantity : new Fraction(spec.quantity),
  });
}

/**
 * Una bolsa inmutable con API fluida.
 *
 * @example
 * const fruta = createBag()
 *   .add({ category: "concreto", type: "comida", subtype: "manzana", quantity: 2 })
 *   .add({ category: "concreto", type: "comida", subtype: "pera", quantity: "1/3" });
 *
 * fruta.entries.length;   // 2 — `fruta` nunca se muta
 * fruta.reset().entries;  // [] — la bolsa vacía es `nulo`
 */
export class ImmutableBag implements BagLiteral {
  readonly type = "BagLiteral" as const;
  readonly entries: readonly Entry[];

  private constructor(entries: readonly Entry[]) {
    this.entries = Object.freeze(entries);
    Object.freeze(this);
  }

  /** @internal — reconstruye una bolsa a partir de entradas ya normalizadas. */
  static of(entries: readonly Entry[]): ImmutableBag {
    return new ImmutableBag(entries);
  }

  /** Devuelve una bolsa nueva con una entrada más al final. */
  add(spec: EntrySpec): ImmutableBag {
    return new ImmutableBag([...this.entries, toEntry(spec)]);
  }

  /**
   * Devuelve la bolsa vacía. No hay `.remove()`: para quitar algo, se parte de
   * `reset()` y se vuelven a añadir las entradas vigentes, lo que evita tener
   * que decidir *cuál* de varias entradas de la misma identidad se quita.
   */
  reset(): ImmutableBag {
    return EMPTY_BAG;
  }
}

const EMPTY_BAG = ImmutableBag.of([]);

/** El único factory de bolsas: empieza vacía y se llena con `.add()`. */
export function createBag(): ImmutableBag {
  return EMPTY_BAG;
}

// =============================================================================
// Criterios (§1.3) — uno por `source`, sin agrupar
// =============================================================================

interface CriterionSpec<P extends string> {
  properties: readonly P[] | P[];
  values: Partial<Record<P, CriterionValue>>;
}

/**
 * Criterio de filtro: un valor único por propiedad, y solo sobre propiedades de
 * identidad (categoría, tipo, subtipo o atributos) — nunca la cantidad.
 */
export function createFilterCriterion<const P extends string>(
  spec: CriterionSpec<P>
): CriterionLiteral<P> {
  return { type: "CriterionLiteral", sourceType: "filter", ...spec };
}

/**
 * Criterio de orden: por propiedad, con dirección `"asc"`/`"desc"` para el orden
 * natural, o una secuencia de valores que fija el orden explícitamente. Puede
 * usar la cantidad (`"quantity"`) además de las propiedades de identidad.
 */
export function createOrderCriterion<const P extends string>(
  spec: CriterionSpec<P>
): CriterionLiteral<P> {
  return { type: "CriterionLiteral", sourceType: "order", ...spec };
}
