// Igualdad — LANGUAGE_SPEC.md §3.2.3

import { denotationallyEqual } from "../runtime/bag";
import type { RuntimeValue } from "../runtime/types";
import { bagAt } from "./helpers";

/**
 * `compare(bolsa, bolsa) → booleano` — binaria.
 *
 * Es la igualdad denotacional del dominio (§1.2.4): compara los vectores, así
 * que ignora el orden, la agrupación y las cantidades 0. Por eso
 * `{ manzana↦0 }` y `nulo` son iguales.
 */
export function compare(args: RuntimeValue[]): RuntimeValue {
  const left = bagAt(args, 0, "compare");
  const right = bagAt(args, 1, "compare");

  return { kind: "booleano", value: denotationallyEqual(left, right) };
}
