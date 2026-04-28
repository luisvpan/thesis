/**
 * Ejecuta programas dataflow en el navegador con @dataflow/interpreter.
 */

import {
  Interpreter,
  type ExecuteResult as InterpreterExecResult,
} from "@dataflow/interpreter";
import type { DataflowProgram as SharedDataflowProgram } from "@dataflow/shared/types";
import { programToSource } from "@/utils/programToSource";
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

const interpreter = new Interpreter();

function getOutputNodeId(program: DataflowProgram): string | undefined {
  return program.graph.nodes.find((n) => n.type === "Output")?.id;
}

/** Interpreta valores del runtime sin importar tipos internos del paquete. */
function runtimeValueToNumber(val: unknown): number | undefined {
  if (!val || typeof val !== "object" || !("kind" in val)) return undefined;
  const v = val as { kind: string; value?: unknown; objectType?: string };

  if (v.kind === "rational" && v.value != null && typeof v.value === "object") {
    const frac = v.value as { valueOf(): number };
    return Number(frac.valueOf?.() ?? frac);
  }

  if (
    v.kind === "abstract" &&
    v.objectType === "rational" &&
    v.value != null &&
    typeof v.value === "object"
  ) {
    const frac = v.value as { valueOf(): number };
    return Number(frac.valueOf?.() ?? frac);
  }

  if (v.kind === "boolean" && typeof v.value === "boolean") {
    return v.value ? 1 : 0;
  }

  return undefined;
}

function formatInterpreterErrors(
  errors: InterpreterExecResult["errors"]
): string {
  const parts = errors.map((e) =>
    e instanceof Error ? e.message : String((e as { message?: string }).message ?? e)
  );
  return parts.length > 0 ? parts.join("; ") : "Error de ejecución";
}

/**
 * Ejecuta un programa dataflow localmente (intérprete incremental en el navegador).
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
  const outputId = getOutputNodeId(program);

  if (!outputId) {
    return { success: false, error: "El programa no tiene salida (Output)" };
  }

  const source = programToSource(program as unknown as SharedDataflowProgram);
  console.log("[execute] JSON programa:", JSON.stringify(program, null, 2));
  console.log("[execute] DSL (intérprete):\n", source);
  if (import.meta.env.DEV) {
    window.__DATAFLOW_LAST_DSL__ = source;
    window.__DATAFLOW_LAST_PROGRAM__ = program;
  }

  try {
    const exec = await interpreter.execute(source);

    if (exec.errors.length > 0) {
      return {
        success: false,
        error: formatInterpreterErrors(exec.errors),
      };
    }

    const raw = exec.results.get(outputId);
    if (raw === undefined) {
      return { success: false, error: "Sin resultado para el nodo de salida" };
    }

    const result = runtimeValueToNumber(raw);
    if (result === undefined || Number.isNaN(result)) {
      return {
        success: false,
        error: "El resultado no es un número usable",
      };
    }

    return { success: true, result };
  } catch (err) {
    console.error("[execute] Error:", err);
    return {
      success: false,
      error: err instanceof Error ? err.message : "Error de ejecución",
    };
  }
}
