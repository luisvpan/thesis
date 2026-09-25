/**
 * El resultado de la última ejecución tal como viaja en `node.data`.
 *
 * Va entero, en un solo campo, y no desmenuzado en campos sueltos: el valor es
 * una unión de cuatro formas y solo una vale a la vez. Repartirlo en campos
 * opcionales dejaba restos de la forma anterior —un arreglo ordenado tapando la
 * descripción de la corrida siguiente—, porque quien lo aplicaba lo esparcía y
 * lo que una forma no nombraba se heredaba.
 */

import type { ResultValue } from "@/services/executeProgram";

/** Lo que una carta con resultado guarda: nada más, y una sola cosa. */
export type WithResultValue = {
  /** El resultado de la última ejecución; ausente si esta carta no tiene. */
  resultValue?: ResultValue;
};

/**
 * La cifra que resume un resultado, para el marcador del operador y el token
 * que viaja por la conexión: el número, o el total de la bolsa. Un booleano y
 * un arreglo de números no se resumen en una cifra.
 */
export function numericValueOf(value: ResultValue | undefined): number | undefined {
  if (!value) return undefined;

  switch (value.kind) {
    case "number":
      return value.value;
    case "semantic":
      return value.result.totalAmount;
    case "boolean":
    case "numberArray":
      return undefined;
  }
}
