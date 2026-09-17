// Presentación de valores para el REPL.
//
// El modo de visualización es decisión del consumidor (§1.5); esto es solo el
// más literal de todos: la notación de la propia especificación.

import type { Bag, Criterion, Entry, RuntimeValue } from "./runtime/types";

export function formatValue(value: RuntimeValue): string {
  switch (value.kind) {
    case "bolsa":
      return formatBag(value);
    case "criterio":
      return formatCriterion(value);
    case "booleano":
      return value.value ? "verdadero" : "falso";
  }
}

/** `{ manzana↦2 [concreto], pera↦3 [concreto] }`; la bolsa vacía es `nulo`. */
function formatBag(value: Bag): string {
  if (value.entries.length === 0) return "nulo";
  return `{ ${value.entries.map(formatEntry).join(", ")} }`;
}

function formatEntry(entry: Entry): string {
  const attributes = Object.entries(entry.attributes);
  const suffix =
    attributes.length > 0 ? ` {${attributes.map(([key, value]) => `${key}: ${value}`).join(", ")}}` : "";

  return `${entry.subtype}↦${entry.quantity} [${entry.category}]${suffix}`;
}

/** `criterio de orden([quantity]) {quantity: "asc"}` */
function formatCriterion(criterion: Criterion): string {
  const values = Object.entries(criterion.values);
  const suffix =
    values.length > 0
      ? ` {${values.map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join(", ")}}`
      : "";

  const subtype = criterion.subtype === "filter" ? "filtro" : "orden";
  return `criterio de ${subtype}([${criterion.properties.join(", ")}])${suffix}`;
}
