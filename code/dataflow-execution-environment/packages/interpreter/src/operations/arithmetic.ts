// Aritmética — LANGUAGE_SPEC.md §3.1

import { aggregate, bag, identityKey, withQuantity } from "../runtime/bag";
import { DataflowError } from "../runtime/errors";
import * as rational from "../runtime/rational";
import type { Entry, RuntimeValue } from "../runtime/types";
import { bagAt, scalarAt } from "./helpers";

/**
 * `sum(bolsa, …) → bolsa` — variádica. Reúne las entradas de todos sus
 * argumentos y las **agrega por identidad**. Es la suma de vectores (§3.1.1).
 *
 * Una identidad cuya suma sea 0 se conserva como entrada de cantidad 0.
 */
export function sum(args: RuntimeValue[]): RuntimeValue {
  const entries: Entry[] = [];

  for (let index = 0; index < args.length; index++) {
    entries.push(...bagAt(args, index, "sum").entries);
  }

  return bag(aggregate(entries));
}

/**
 * `substract(bolsa, bolsa) → bolsa` — binaria. Resta por identidad; es la resta
 * de vectores (§3.1.2). Las identidades presentes solo en el sustraendo quedan
 * con cantidad negativa, y las que resulten 0 se conservan.
 */
export function substract(args: RuntimeValue[]): RuntimeValue {
  const minuend = aggregate(bagAt(args, 0, "substract").entries);
  const subtrahend = aggregate(bagAt(args, 1, "substract").entries);

  const remaining = new Map(subtrahend.map((entry) => [identityKey(entry), entry]));
  const result: Entry[] = [];

  for (const entry of minuend) {
    const key = identityKey(entry);
    const other = remaining.get(key);
    remaining.delete(key);
    result.push(
      other ? withQuantity(entry, rational.subtract(entry.quantity, other.quantity)) : entry
    );
  }

  for (const entry of remaining.values()) {
    result.push(withQuantity(entry, rational.subtract(rational.zero(), entry.quantity)));
  }

  return bag(result);
}

/**
 * `multiply(bolsa, número) → bolsa` — binaria. La posición desambigua: el
 * segundo argumento es siempre el escalar (§3.1.3). Opera **entrada por
 * entrada** y conserva los repetidos.
 */
export function multiply(args: RuntimeValue[]): RuntimeValue {
  const value = bagAt(args, 0, "multiply");
  const scalar = scalarAt(args, 1, "multiply", "escalar");

  if (scalar === null) return value;

  return bag(value.entries.map((entry) => withQuantity(entry, rational.multiply(entry.quantity, scalar))));
}

/**
 * `divide(bolsa, número) → bolsa` — binaria. El segundo argumento es el divisor
 * (§3.1.4). Como `multiply`, opera entrada por entrada.
 */
export function divide(args: RuntimeValue[]): RuntimeValue {
  const value = bagAt(args, 0, "divide");
  const divisor = scalarAt(args, 1, "divide", "divisor");

  if (divisor === null) return value;

  if (rational.isZero(divisor)) {
    throw new DataflowError("DIVISION_BY_ZERO", "divide no admite el divisor 0", { argumentIndex: 1 });
  }

  return bag(value.entries.map((entry) => withQuantity(entry, rational.divide(entry.quantity, divisor))));
}
