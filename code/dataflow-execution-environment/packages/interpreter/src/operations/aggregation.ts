// Agregación — LANGUAGE_SPEC.md §3.6.1

import { numberBag } from "../runtime/bag";
import * as rational from "../runtime/rational";
import type { RuntimeValue } from "../runtime/types";
import { bagAt } from "./helpers";

/**
 * `count(bolsa) → número` — unaria. Totaliza sin importar la identidad: no
 * agrupa ni distingue por tipo, solo suma cantidades. Sobre `nulo` da 0.
 *
 * Como el resultado es un número, puede alimentar a las operaciones que esperan
 * uno (el escalar de `multiply`, el umbral de `less_than`…).
 */
export function count(args: RuntimeValue[]): RuntimeValue {
  let total = rational.zero();

  for (const entry of bagAt(args, 0, "count").entries) {
    total = rational.add(total, entry.quantity);
  }

  return numberBag(total);
}
