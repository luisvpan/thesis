// Pretty-print utilities for RuntimeValue types

import type { RuntimeValue, CPAObject, ArrayValue } from "./runtime/types";

/**
 * Formats a RuntimeValue for display in the REPL
 */
export function formatValue(value: RuntimeValue): string {
  switch (value.kind) {
    case "racional":
      return value.value.toString();
    case "booleano":
      return value.value ? "true" : "false";
    case "cpa":
      return formatCPAObject(value);
    case "arreglo":
      return formatArray(value);
    case "otro":
      return `"${value.value}"`;
    default:
      return JSON.stringify(value);
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
