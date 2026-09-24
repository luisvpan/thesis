// Aritmética — LANGUAGE_SPEC.md §3.1

import { aggregate, bag, identityKey, withQuantity } from "../runtime/bag";
import { DataflowError } from "../runtime/errors";
import * as rational from "../runtime/rational";
import type { Bag, Entry, RuntimeValue } from "../runtime/types";
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

  return bag(aggregate({ entries }));
}

/**
 * `substract(bolsa, bolsa) → bolsa` — binaria. Resta por identidad; es la resta
 * de vectores (§3.1.2). Las identidades presentes solo en el sustraendo quedan
 * con cantidad negativa, y las que resulten 0 se conservan.
 */
export function substract(args: RuntimeValue[]): RuntimeValue {
  const minuend = aggregate({ entries: bagAt(args, 0, "substract").entries });
  const subtrahend = aggregate({ entries: bagAt(args, 1, "substract").entries });

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
 * La bolsa a escalar, en forma reducida. `multiply` y `divide` **fabrican**
 * cantidades, así que emiten la forma canónica: la agrupación que traen sus
 * argumentos es la que el niño puso sobre la mesa, y escalarla entrada por
 * entrada inventaría pilas que nadie colocó —seis medias manzanas donde hay
 * tres— que luego `first` y `last` leerían como si fueran reales (§3.1.3-4).
 *
 * Lo abstracto queda fuera de la agrupación: cada carta de número es una unidad
 * aparte, y fundirlas daría `multiply({7,2,5}, 2) = {28}` (§3, convenciones).
 */
function scalable(value: Bag): Entry[] {
  return aggregate({ entries: value.entries, keep: "abstracto" });
}

/**
 * `multiply(bolsa, número) → bolsa` — binaria. La posición desambigua: el
 * segundo argumento es siempre el escalar (§3.1.3).
 */
export function multiply(args: RuntimeValue[]): RuntimeValue {
  const value = bagAt(args, 0, "multiply");
  const scalar = scalarAt(args, 1, "multiply", "escalar");

  if (scalar === null) return value;

  return bag(scalable(value).map((entry) => withQuantity(entry, rational.multiply(entry.quantity, scalar))));
}

/**
 * `divide(bolsa, número) → bolsa` — binaria. El segundo argumento es el divisor
 * (§3.1.4). Como `multiply`, agrupa antes de escalar.
 */
export function divide(args: RuntimeValue[]): RuntimeValue {
  const value = bagAt(args, 0, "divide");
  const divisor = scalarAt(args, 1, "divide", "divisor");

  if (divisor === null) return value;

  if (rational.isZero(divisor)) {
    throw new DataflowError("DIVISION_BY_ZERO", "divide no admite el divisor 0", { argumentIndex: 1 });
  }

  return bag(scalable(value).map((entry) => withQuantity(entry, rational.divide(entry.quantity, divisor))));
}
