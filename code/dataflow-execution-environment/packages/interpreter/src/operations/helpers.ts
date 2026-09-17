// Utilidades comunes a las operaciones.
//
// La aridad y la categoría de cada argumento las garantiza la pasada estática
// (§4.2.5-7); lo que aquí se comprueba es lo que solo se sabe con el valor: si
// una bolsa es además un número (§4.3.1).

import type Fraction from "fraction.js";
import { NULO, asNumber, isNulo } from "../runtime/bag";
import { isCriterionComplete } from "../runtime/criteria";
import { RuntimeError } from "../runtime/errors";
import type { Bag, Criterion, RuntimeValue } from "../runtime/types";
import { isBag, isCriterion } from "../runtime/types";

/**
 * La bolsa de una posición. Un argumento ausente se comporta como `nulo`, que
 * toda operación ignora (§1.2.5).
 */
export function bagAt(args: RuntimeValue[], index: number, operation: string): Bag {
  const value = args[index];
  if (value === undefined) return NULO;
  if (!isBag(value)) {
    throw new RuntimeError(
      "TYPE_ERROR",
      `${operation} espera una bolsa en la posición ${index + 1}, y recibió un ${value.kind}`,
      { argumentIndex: index }
    );
  }
  return value;
}

/**
 * El valor del número que acompaña a una operación (el escalar de `multiply` y
 * `divide`, el umbral de `less_than` y `greater_than`).
 *
 * Devuelve `null` cuando el argumento es `nulo`: la regla noop manda sobre el
 * error, así que la operación devuelve su bolsa sin tocar (§1.2.5). Una bolsa
 * no vacía que no es un número sí es un error de ejecución (§4.3.1).
 */
export function scalarAt(
  args: RuntimeValue[],
  index: number,
  operation: string,
  role: string
): Fraction | null {
  const value = bagAt(args, index, operation);
  if (isNulo(value)) return null;

  const number = asNumber(value);
  if (number === null) {
    throw new RuntimeError(
      "EXPECTED_NUMBER",
      `${operation} espera un número como ${role}, y recibió una bolsa que no lo es`,
      { argumentIndex: index }
    );
  }

  return number;
}

/**
 * Los criterios completos que siguen a la bolsa. Los incompletos se descartan
 * (§3.3.1 y §3.4.1, paso 1); los criterios no se agrupan, así que cada uno
 * llega en su propio argumento (§1.3).
 */
export function criteriaFrom(args: RuntimeValue[], from: number): Criterion[] {
  return args.slice(from).filter(isCriterion).filter(isCriterionComplete);
}
