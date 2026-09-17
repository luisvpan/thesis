// Criterios — LANGUAGE_SPEC.md §1.3
//
// Un criterio declara su subtipo (`filter` u `order`), y el subtipo determina
// cómo se leen sus valores. Lo que aquí se decide es qué criterios están
// completos y cómo se lee una propiedad de una entrada; que la forma de los
// valores corresponda al subtipo lo verifica la pasada estática (§4.2.7).

import type Fraction from "fraction.js";
import type { Criterion, CriterionValue, Entry, OrderDirection } from "./types";

/** La única propiedad que no es de identidad; solo el criterio de orden la usa. */
export const QUANTITY_PROPERTY = "quantity";

export function isDirection(value: CriterionValue): value is OrderDirection {
  return value === "asc" || value === "desc";
}

export function isSequence(value: CriterionValue): value is string[] {
  return Array.isArray(value);
}

/**
 * Un criterio de filtro está completo si fija un valor único para cada una de
 * sus propiedades; uno de orden, si fija para cada una una dirección o una
 * secuencia no vacía. Los incompletos se descartan (§3.3.1 y §3.4.1, paso 1).
 */
export function isCriterionComplete(criterion: Criterion): boolean {
  if (criterion.properties.length === 0) return false;

  return criterion.properties.every((property) => {
    const value = criterion.values[property];
    if (value === undefined) return false;

    return criterion.subtype === "filter"
      ? typeof value === "string" && value.length > 0
      : isDirection(value) || (isSequence(value) && value.length > 0);
  });
}

/**
 * El valor textual de una propiedad de identidad de una entrada: categoría,
 * tipo, subtipo o uno de sus atributos (§1.2.1).
 */
export function entryText(entry: Entry, property: string): string | undefined {
  switch (property) {
    case "category":
      return entry.category;
    case "type":
      return entry.type;
    case "subtype":
      return entry.subtype;
    default:
      return entry.attributes[property];
  }
}

/** La cantidad, cuando la propiedad pedida es la cantidad; si no, `undefined`. */
export function entryQuantity(entry: Entry, property: string): Fraction | undefined {
  return property === QUANTITY_PROPERTY ? entry.quantity : undefined;
}
