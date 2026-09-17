// El álgebra de la bolsa — LANGUAGE_SPEC.md §1.2
//
// Aquí vive todo lo que las operaciones necesitan saber sobre bolsas: la clave
// de identidad, la agregación (bolsa → forma reducida), la denotación (bolsa →
// vector) y el papel de escalar de los números.

import type Fraction from "fraction.js";
import * as rational from "./rational";
import type { Bag, Entry } from "./types";

/** La bolsa vacía: el vector cero (§1.2.5). */
export const NULO: Bag = Object.freeze({ kind: "bolsa", entries: Object.freeze([]) }) as Bag;

export function bag(entries: readonly Entry[]): Bag {
  return entries.length === 0 ? NULO : { kind: "bolsa", entries };
}

/** Un argumento `nulo` se trata como ausente por toda operación (§1.2.5). */
export function isNulo(value: Bag): boolean {
  return value.entries.length === 0;
}

/**
 * Clave de la identidad CPA: categoría, tipo, subtipo y **todos** los atributos
 * (§1.2.1). Se serializa con JSON para que ningún valor pueda colisionar con el
 * separador.
 */
export function identityKey(entry: Entry): string {
  const attributes = Object.entries(entry.attributes).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return JSON.stringify([entry.category, entry.type, entry.subtype, attributes]);
}

export function sameIdentity(a: Entry, b: Entry): boolean {
  return identityKey(a) === identityKey(b);
}

export function withQuantity(entry: Entry, quantity: Fraction): Entry {
  return { ...entry, attributes: { ...entry.attributes }, quantity };
}

/**
 * Forma reducida: una entrada por identidad, en orden de primera aparición, con
 * la suma de sus cantidades. Las cantidades 0 **se conservan** (§1.2.2c).
 */
export function aggregate(entries: readonly Entry[]): Entry[] {
  const groups = new Map<string, Entry>();

  for (const entry of entries) {
    const key = identityKey(entry);
    const previous = groups.get(key);
    groups.set(
      key,
      previous
        ? withQuantity(previous, rational.add(previous.quantity, entry.quantity))
        : withQuantity(entry, entry.quantity)
    );
  }

  return Array.from(groups.values());
}

export function aggregated(value: Bag): Bag {
  return bag(aggregate(value.entries));
}

/**
 * Denotación: el vector en ℚ^(Id) (§1.2.3). Agrega por identidad y **descarta
 * el soporte nulo**, porque una cantidad 0 no aporta nada al vector. Es la base
 * de la igualdad denotacional (§1.2.4).
 */
export function denote(value: Bag): Map<string, Fraction> {
  const vector = new Map<string, Fraction>();

  for (const entry of aggregate(value.entries)) {
    if (!entry.quantity.equals(rational.zero())) {
      vector.set(identityKey(entry), entry.quantity);
    }
  }

  return vector;
}

/** Igualdad denotacional: ignora orden, agrupación y cantidades 0 (§1.2.4). */
export function denotationallyEqual(a: Bag, b: Bag): boolean {
  const left = denote(a);
  const right = denote(b);

  if (left.size !== right.size) return false;

  for (const [key, quantity] of left) {
    const other = right.get(key);
    if (!other || !rational.equals(quantity, other)) return false;
  }

  return true;
}

// =============================================================================
// Números (§1.2.6)
// =============================================================================

export function isNumberEntry(entry: Entry): boolean {
  return entry.category === "abstracto" && entry.type === "numero";
}

/** El número `n`: la bolsa `{ (abstracto, numero, racional)↦n }`. */
export function numberBag(quantity: Fraction): Bag {
  return bag([
    {
      category: "abstracto",
      type: "numero",
      subtype: "racional",
      attributes: {},
      quantity,
    },
  ]);
}

/**
 * El valor numérico de una bolsa, o `null` si no es un número.
 *
 * Una bolsa es un número cuando, agregada por identidad, toda su entrada es
 * numérica; su valor es la suma de las cantidades. Así `count(nulo)` —el número
 * 0, cuyo soporte es vacío— sigue sirviendo de escalar, mientras que
 * `{ manzana↦0 }` no es un número.
 *
 * La bolsa vacía devuelve `null`: `nulo` no es "el número 0", es un argumento
 * ausente, y quien la reciba aplica la regla noop (§1.2.5).
 */
export function asNumber(value: Bag): Fraction | null {
  if (value.entries.length === 0) return null;

  let total = rational.zero();
  for (const entry of value.entries) {
    if (!isNumberEntry(entry)) return null;
    total = rational.add(total, entry.quantity);
  }

  return total;
}
