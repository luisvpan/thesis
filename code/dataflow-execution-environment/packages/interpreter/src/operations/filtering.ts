// Filtrado — LANGUAGE_SPEC.md §3.4.1

import { bag } from "../runtime/bag";
import { entryText } from "../runtime/criteria";
import type { Criterion, Entry, RuntimeValue } from "../runtime/types";
import { bagAt, criteriaFrom } from "./helpers";

/**
 * Una entrada satisface un criterio si cumple **todas** sus restricciones
 * `propiedad = valor` (Y entre propiedades), cada una con un valor único sobre
 * una propiedad de identidad.
 *
 * Un valor múltiple, o una restricción sobre la cantidad, es un criterio
 * inadecuado: lo rechaza la pasada estática (§4.2.7), y aquí simplemente no
 * casa con nada.
 */
function satisfies(entry: Entry, criterion: Criterion): boolean {
  return criterion.properties.every((property) => {
    const expected = criterion.values[property];
    return typeof expected === "string" && entryText(entry, property) === expected;
  });
}

/**
 * `filter(bolsa, criterio, …) → bolsa` — variádica en forma normal disyuntiva:
 * Y entre las propiedades de un criterio, O entre criterios distintos.
 *
 * Trabaja entrada por entrada y conserva los repetidos: como el criterio prueba
 * la identidad y no la cantidad, los de una misma identidad pasan o se
 * descartan todos juntos (§3.4.1, nota).
 */
export function filter(args: RuntimeValue[]): RuntimeValue {
  const value = bagAt(args, 0, "filter");
  const criteria = criteriaFrom(args, 1);

  if (criteria.length === 0) return value;

  return bag(value.entries.filter((entry) => criteria.some((criterion) => satisfies(entry, criterion))));
}
