/**
 * Servicio para ejecutar programas dataflow en el backend.
 */

import {
  serializeProgram,
  type DataflowProgram,
} from "@/utils/serializeProgram";
import type { DataflowNode } from "@/contexts/NodeContext";
import type { Edge } from "@xyflow/react";

export type ExecuteResult = {
  success: boolean;
  result?: number;
  error?: string;
};

/**
 * Ejecuta un programa dataflow en el backend.
 * Serializa los nodos y edges, envía al endpoint /execute, y retorna el resultado.
 */
export async function executeProgram(
  nodes: DataflowNode[],
  edges: Edge[],
  programOverride?: DataflowProgram
): Promise<ExecuteResult> {
  if (nodes.length === 0 && !programOverride) {
    return { success: false, error: "No hay nodos para ejecutar" };
  }

  const program = programOverride ?? serializeProgram(nodes, edges);

  console.log("[execute] Programa serializado:", JSON.stringify(program, null, 2));

  try {
    const response = await fetch("/api/v1/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ program }),
    });

    const data = await response.json();

    console.log("[execute] Respuesta del backend:", data);

    if (!data.success) {
      const errorMsg = data.errors?.[0]?.message || data.error || "Error de ejecución";
      return { success: false, error: errorMsg };
    }

    // El resultado está en outputs[0]
    // Puede ser un objeto { kind: "natural", value: X } o directamente el valor
    const output = data.outputs?.[0];
    let result: number | undefined;

    if (output === undefined || output === null) {
      return { success: false, error: "Sin resultado" };
    }

    if (typeof output === "object" && "value" in output) {
      result = output.value;
    } else if (typeof output === "number") {
      result = output;
    } else {
      result = Number(output);
    }

    return { success: true, result };
  } catch (err) {
    console.error("[execute] Error:", err);
    return {
      success: false,
      error: err instanceof Error ? err.message : "Error de red",
    };
  }
}
