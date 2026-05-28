// Pretty-print utilities for RuntimeValue types

import type { RuntimeValue, CPAObject, ArrayValue, CriteriaObject } from "./runtime/types";

/**
 * Formats a RuntimeValue for display in the REPL
 */
export function formatValue(value: RuntimeValue): string {
  switch (value.kind) {
    case "cpa":
      return formatCPAObject(value);
    case "arreglo":
      return formatArray(value);
    case "criteria":
      return formatCriteria(value);
    case "booleano":
      return value.value ? "verdadero" : "falso";
    case "otro":
      return `"${value.value}"`;
  }
}

/**
 * Formats a CPA object for display
 * Example: circulo(3) [pictorico] {size: grande}
 */
function formatCPAObject(obj: CPAObject): string {
  const attrs = Object.entries(obj.attributes);
  const attrStr = attrs.length > 0
    ? ` {${attrs.map(([k, v]) => `${k}: ${v}`).join(", ")}}`
    : "";

  return `${obj.subtype}(${obj.quantity}) [${obj.category}]${attrStr}`;
}

/**
 * Formats an array for display
 */
function formatArray(arr: ArrayValue): string {
  return `[${arr.elements.map(formatValue).join(", ")}]`;
}

/**
 * Formats a criteria object for display
 * Example: criteria([size, color]) {size: "grande"}
 */
function formatCriteria(obj: CriteriaObject): string {
  const values = Object.entries(obj.values);
  const valuesStr = values.length > 0
    ? ` {${values.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(", ")}}`
    : "";

  return `criteria([${obj.properties.join(", ")}])${valuesStr}`;
}
