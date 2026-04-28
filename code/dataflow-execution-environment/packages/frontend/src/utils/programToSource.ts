/**
 * DSL textual compatible con **@dataflow/interpreter** (Chevrotain).
 *
 * Gramática esperada (resumen):
 * - `source id = número;`
 * - `transform id = sum|substract|multiply|divide(...);`
 * - `sink id = id_transform_o_fuente;`
 *
 * No usar la forma del compilador (`source id: natural = …`, `output …`) porque el intérprete la rechaza.
 */

import type {
  DataflowProgram,
  DataSourceNode,
  TransformationNode,
  OutputNode,
} from "@dataflow/shared/types";

/** Operadores del grafo serializado → palabras clave del intérprete. */
const TRANSFORM_OP_TO_INTERPRETER: Record<string, string> = {
  ADD: "sum",
  SUBTRACT: "substract",
  MULTIPLY: "multiply",
  DIVIDE: "divide",
};

export function programToSource(program: DataflowProgram): string {
  const lines: string[] = [];
  const { nodes } = program.graph;

  for (const node of nodes) {
    if (node.type === "DataSource") {
      const src = node as DataSourceNode;
      const num = extractNumericLiteral(src.value);
      lines.push(`source ${src.id} = ${num};`);
    }
  }

  for (const node of nodes) {
    if (node.type === "Transformation") {
      const tr = node as TransformationNode;
      const op =
        TRANSFORM_OP_TO_INTERPRETER[tr.operation] ?? String(tr.operation).toLowerCase();
      const inputsStr = tr.inputs.join(", ");
      lines.push(`transform ${tr.id} = ${op}(${inputsStr});`);
    }
  }

  for (const node of nodes) {
    if (node.type === "Output") {
      const out = node as OutputNode;
      lines.push(`sink ${out.id} = ${out.input};`);
    }
  }

  return lines.join("\n");
}

function extractNumericLiteral(value: unknown): string {
  if (value === null || value === undefined) return "0";
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value === "object" && "kind" in value && "value" in value) {
    const v = (value as { value: unknown }).value;
    if (typeof v === "number" && Number.isFinite(v)) return String(v);
  }
  return "0";
}
