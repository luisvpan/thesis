// Utilidades compartidas por las pruebas.

import Fraction from "fraction.js";
import { bag, numberBag } from "../runtime/bag";
import type { Bag, CPACategory, Criterion, CriterionValue, Entry, RuntimeValue } from "../runtime/types";

type Quantity = number | string | Fraction;

export function entry(
  subtype: string,
  quantity: Quantity,
  attributes: Record<string, string> = {},
  identity: { category?: CPACategory; type?: string } = {}
): Entry {
  return {
    category: identity.category ?? "concreto",
    type: identity.type ?? "comida",
    subtype,
    attributes,
    quantity: quantity instanceof Fraction ? quantity : new Fraction(quantity),
  };
}

export function bagOf(...entries: Entry[]): Bag {
  return bag(entries);
}

/** El número `n`: la bolsa `{ (abstracto, numero, racional)↦n }`. */
export function num(quantity: Quantity): Bag {
  return numberBag(quantity instanceof Fraction ? quantity : new Fraction(quantity));
}

export function filterCriterion(
  properties: string[],
  values: Record<string, CriterionValue>
): Criterion {
  return { kind: "criterio", subtype: "filter", properties, values };
}

export function orderCriterion(
  properties: string[],
  values: Record<string, CriterionValue>
): Criterion {
  return { kind: "criterio", subtype: "order", properties, values };
}

/**
 * La bolsa como lista de `subtipo:cantidad`, en su orden: es lo que se compara
 * en casi todas las pruebas, porque conserva orden, repetidos y ceros.
 */
export function pairs(value: RuntimeValue): string[] {
  if (value.kind !== "bolsa") throw new Error(`Se esperaba una bolsa, y llegó un ${value.kind}`);
  return value.entries.map((item) => `${item.subtype}:${item.quantity.toFraction()}`);
}

/** Un número escrito en la sintaxis del lenguaje, para las pruebas por texto. */
export function numberLiteral(quantity: number | string): string {
  return `{"sourceType": "data", "category": "abstracto", "type": "numero", "subtype": "racional", "quantity": ${quantity}}`;
}

/** Un objeto de datos escrito en la sintaxis del lenguaje. */
export function dataLiteral(
  category: CPACategory,
  type: string,
  subtype: string,
  quantity: number | string,
  attributes: Record<string, string> = {}
): string {
  const extra = Object.entries(attributes).map(([key, value]) => `, "${key}": "${value}"`);
  return `{"sourceType": "data", "category": "${category}", "type": "${type}", "subtype": "${subtype}", "quantity": ${quantity}${extra.join("")}}`;
}

/** El valor de una bolsa de una sola entrada, como fracción exacta. */
export function only(value: RuntimeValue): string {
  const list = pairs(value);
  if (list.length !== 1) throw new Error(`Se esperaba una entrada, y hay ${list.length}`);
  return value.kind === "bolsa" ? value.entries[0].quantity.toFraction() : "";
}
