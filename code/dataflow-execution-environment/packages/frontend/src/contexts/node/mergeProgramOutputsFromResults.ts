import type { OutputErrorInfo, ResultValue } from "@/services/executeProgram";
import { logger } from "@/lib/logger";
import { safeJsonStringify } from "@/utils/jsonReplacer";
import type { WithResultValue } from "@/utils/resultValueDisplay";
import type { DataflowNode } from "./types";

function sameErrors(current: OutputErrorInfo[] | undefined, next: OutputErrorInfo[]): boolean {
  return safeJsonStringify(current ?? []) === safeJsonStringify(next);
}

function sameResult(current: ResultValue | undefined, next: ResultValue | undefined): boolean {
  return safeJsonStringify(current ?? null) === safeJsonStringify(next ?? null);
}

/**
 * Aplica resultados del intérprete a nodos `programOutput` y `operator`, y deja
 * en cada carta de salida los errores que le tocan: un error de una salida no es
 * asunto de las demás (§4).
 *
 * El resultado se reemplaza entero. Una carta que esta vez no trajo ninguno
 * —porque le quitaron la entrada, o porque su camino se apagó— se queda sin él,
 * en vez de seguir mostrando el de la corrida anterior.
 */
export function mergeProgramOutputsFromResults(
  nodes: DataflowNode[],
  results: Map<string, ResultValue>,
  errorsByOutput: Map<string, OutputErrorInfo[]> = new Map()
): DataflowNode[] {
  try {
    let changed = false;

    const updated = nodes.map((n) => {
      if (n.type !== "programOutput" && n.type !== "operator") return n;

      const currentData = n.data as WithResultValue & { errors?: OutputErrorInfo[] };
      const resultValue = results.get(n.id);

      // Los errores solo cuelgan de las cartas de salida.
      const errors = n.type === "programOutput" ? (errorsByOutput.get(n.id) ?? []) : [];
      const errorsChanged = n.type === "programOutput" && !sameErrors(currentData.errors, errors);

      if (sameResult(currentData.resultValue, resultValue) && !errorsChanged) return n;

      changed = true;
      // `data` es una unión discriminada por el tipo de nodo y esparcirla la
      // aplana; el resultado es común a las dos, así que se reafirma.
      return {
        ...n,
        data: {
          ...n.data,
          resultValue,
          ...(n.type === "programOutput" ? { errors } : {}),
        },
      } as DataflowNode;
    });

    return changed ? updated : nodes;
  } catch (err) {
    logger.nodeProvider.error("Failed to merge program outputs", {
      error: err instanceof Error ? err.message : String(err),
    });
    return nodes;
  }
}
