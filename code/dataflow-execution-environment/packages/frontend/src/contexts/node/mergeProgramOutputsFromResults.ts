import type { OperatorFlowNodeData } from "@/components/dataflow";
import type { ResultValue } from "@/services/executeProgram";
import {
  resultValueToDisplayData,
  type FlowResultDisplayData,
} from "@/utils/evalResultDisplay";
import { logger } from "@/lib/logger";
import { safeJsonStringify } from "@/utils/jsonReplacer";
import type { DataflowNode } from "./types";

function displayDataUnchanged(
  current: FlowResultDisplayData,
  next: FlowResultDisplayData
): boolean {
  return (
    current.value === next.value &&
    current.description === next.description &&
    safeJsonStringify(current.visualStrip) === safeJsonStringify(next.visualStrip) &&
    current.isSingleCpaObject === next.isSingleCpaObject &&
    safeJsonStringify(current.singleCpaObjectMeta) ===
      safeJsonStringify(next.singleCpaObjectMeta) &&
    current.numerator === next.numerator &&
    current.denominator === next.denominator &&
    current.booleanValue === next.booleanValue &&
    safeJsonStringify(current.numberArrayValues) ===
      safeJsonStringify(next.numberArrayValues)
  );
}

/** Aplica resultados del intérprete a nodos `programOutput` y `operator`. */
export function mergeProgramOutputsFromResults(
  nodes: DataflowNode[],
  results: Map<string, ResultValue>
): DataflowNode[] {
  try {
    let changed = false;
    const updated = nodes.map((n) => {
      if (n.type !== "programOutput" && n.type !== "operator") return n;
      const resultValue = results.get(n.id);
      if (resultValue === undefined) return n;

      const newData = resultValueToDisplayData(resultValue);
      const currentData = n.data as FlowResultDisplayData & OperatorFlowNodeData;

      if (displayDataUnchanged(currentData, newData)) {
        return n;
      }

      changed = true;
      if (n.type === "operator") {
        const opData = n.data as OperatorFlowNodeData;
        // Determine result value for edge display
        let operatorResult: number | undefined = newData.value;
        if (operatorResult === undefined) {
          if (resultValue.kind === "boolean") {
            operatorResult = resultValue.value ? 1 : 0;
          } else if (resultValue.kind === "semantic") {
            operatorResult = resultValue.result.totalAmount;
          } else if (resultValue.kind === "numberArray") {
            // For number arrays, show count
            operatorResult = resultValue.values.length;
          }
        }
        return {
          ...n,
          data: {
            ...opData,
            ...newData,
            result: operatorResult,
          },
        };
      }

      return { ...n, data: { ...n.data, ...newData } };
    });
    return changed ? updated : nodes;
  } catch (err) {
    logger.nodeProvider.error("Failed to merge program outputs", {
      error: err instanceof Error ? err.message : String(err),
    });
    return nodes;
  }
}
