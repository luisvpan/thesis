// Orden — LANGUAGE_SPEC.md §3.3.1
//
// Una sola operación: la dirección vive en el criterio, no en el nombre.

import { aggregate, bag } from "../runtime/bag";
import { QUANTITY_PROPERTY, entryText, isDirection, isSequence } from "../runtime/criteria";
import * as rational from "../runtime/rational";
import type { Criterion, CriterionValue, Entry, RuntimeValue } from "../runtime/types";
import { bagAt, criteriaFrom } from "./helpers";

interface SortKey {
  property: string;
  value: CriterionValue;
}

/** Las claves de ordenamiento, de izquierda a derecha: la primera manda. */
function sortKeys(criteria: Criterion[]): SortKey[] {
  const keys: SortKey[] = [];

  for (const criterion of criteria) {
    for (const property of criterion.properties) {
      keys.push({ property, value: criterion.values[property]! });
    }
  }

  return keys;
}

/** La posición de una entrada en una secuencia explícita; los ausentes, al final. */
function sequencePosition(entry: Entry, property: string, sequence: string[]): number {
  const text = entryText(entry, property);
  const position = text === undefined ? -1 : sequence.indexOf(text);
  return position === -1 ? Number.POSITIVE_INFINITY : position;
}

function compareByKey(a: Entry, b: Entry, key: SortKey): number {
  if (isSequence(key.value)) {
    // La secuencia *es* el orden, así que no lleva dirección. Dos entradas fuera
    // de la secuencia empatan (y las desempata el orden previo).
    const left = sequencePosition(a, key.property, key.value);
    const right = sequencePosition(b, key.property, key.value);
    if (left === right) return 0;
    return left < right ? -1 : 1;
  }

  if (!isDirection(key.value)) return 0;

  const comparison =
    key.property === QUANTITY_PROPERTY
      ? // La cantidad se compara numéricamente, no como texto.
        rational.compare(a.quantity, b.quantity)
      : (entryText(a, key.property) ?? "").localeCompare(entryText(b, key.property) ?? "");

  return key.value === "asc" ? comparison : -comparison;
}

/**
 * `order(bolsa, criterio, …) → bolsa`.
 *
 * Descarta los criterios incompletos; si no queda ninguno, devuelve la bolsa sin
 * cambios. Agrupa por identidad y ordena aplicando los criterios en orden: el
 * primero manda y los siguientes desempatan. El orden es estable.
 */
export function order(args: RuntimeValue[]): RuntimeValue {
  const value = bagAt(args, 0, "order");
  const keys = sortKeys(criteriaFrom(args, 1));

  if (keys.length === 0) return value;

  const decorated = aggregate(value.entries).map((entry, index) => ({ entry, index }));

  decorated.sort((a, b) => {
    for (const key of keys) {
      const comparison = compareByKey(a.entry, b.entry, key);
      if (comparison !== 0) return comparison;
    }
    return a.index - b.index;
  });

  return bag(decorated.map(({ entry }) => entry));
}
