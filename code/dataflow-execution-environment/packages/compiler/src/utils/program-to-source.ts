import type { DataflowProgram, DataSourceNode, TransformationNode, OutputNode } from "@dataflow/shared/types";

/**
 * Convierte un DataflowProgram al formato de texto fuente DSL
 * para poder compilarlo con compiler.compile() y obtener validación completa.
 */
export function programToSource(program: DataflowProgram): string {
  const lines: string[] = [];
  const { nodes } = program.graph;

  // 1. DataSource → source statement
  for (const node of nodes) {
    if (node.type === "DataSource") {
      const src = node as DataSourceNode;
      const valueStr = serializeValue(src.value, src.dataType);
      lines.push(`source ${src.id}: ${src.dataType} = ${valueStr};`);
    }
  }

  // 2. Transformation → transform statement
  for (const node of nodes) {
    if (node.type === "Transformation") {
      const tr = node as TransformationNode;
      const inputsStr = tr.inputs.join(", ");
      lines.push(`transform ${tr.id}: ${tr.dataType} = ${tr.operation}(${inputsStr});`);
    }
  }

  // 3. Output → output statement
  for (const node of nodes) {
    if (node.type === "Output") {
      const out = node as OutputNode;
      lines.push(`output ${out.id}: ${out.dataType} = ${out.input};`);
    }
  }

  return lines.join("\n");
}

function serializeValue(value: unknown, dataType: string): string {
  if (value === null || value === undefined) {
    return "0";
  }

  // Valor directo (número)
  if (typeof value === "number") {
    return String(value);
  }

  // Objeto con kind/value
  if (typeof value === "object" && "kind" in value && "value" in value) {
    const v = (value as { kind: string; value: unknown }).value;
    if (typeof v === "number") return String(v);
    if (typeof v === "string") return `"${v}"`;
    if (typeof v === "boolean") return String(v);
  }

  // Fallback
  return JSON.stringify(value);
}
