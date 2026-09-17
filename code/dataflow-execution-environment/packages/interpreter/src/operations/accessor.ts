// Acceso posicional — LANGUAGE_SPEC.md §3.5
//
// Leen el orden vigente y trabajan entrada por entrada: sobre una bolsa con
// repetidos de una misma identidad señalan una pila individual, no su total.

import { NULO, bag } from "../runtime/bag";
import type { RuntimeValue } from "../runtime/types";
import { bagAt } from "./helpers";

/** `first(bolsa) → bolsa` — unaria (§3.5.1). */
export function first(args: RuntimeValue[]): RuntimeValue {
  const entries = bagAt(args, 0, "first").entries;
  return entries.length === 0 ? NULO : bag([entries[0]]);
}

/** `last(bolsa) → bolsa` — unaria (§3.5.2). */
export function last(args: RuntimeValue[]): RuntimeValue {
  const entries = bagAt(args, 0, "last").entries;
  return entries.length === 0 ? NULO : bag([entries[entries.length - 1]]);
}
