// Comparación con umbral — LANGUAGE_SPEC.md §3.2.1 y §3.2.2

import type Fraction from "fraction.js";
import { aggregate, bag } from "../runtime/bag";
import * as rational from "../runtime/rational";
import type { RuntimeValue } from "../runtime/types";
import { bagAt, scalarAt } from "./helpers";

/**
 * Conserva las identidades cuya cantidad **total** cumple la comparación con el
 * umbral. Agrupa por identidad antes de comparar, para que el resultado dependa
 * solo del vector: `{ manzana↦2, manzana↦3 }` y `{ manzana↦5 }` se comparan
 * igual (§3.2.1, nota).
 */
function threshold(
  args: RuntimeValue[],
  operation: "less_than" | "greater_than",
  keep: (quantity: Fraction, limit: Fraction) => boolean
): RuntimeValue {
  const value = bagAt(args, 0, operation);
  const limit = scalarAt(args, 1, operation, "umbral");

  if (limit === null) return value;

  return bag(aggregate(value.entries).filter((entry) => keep(entry.quantity, limit)));
}

/** `less_than(bolsa, número) → bolsa` — binaria (§3.2.1). */
export function lessThan(args: RuntimeValue[]): RuntimeValue {
  return threshold(args, "less_than", rational.lessThan);
}

/** `greater_than(bolsa, número) → bolsa` — binaria (§3.2.2). */
export function greaterThan(args: RuntimeValue[]): RuntimeValue {
  return threshold(args, "greater_than", rational.greaterThan);
}
