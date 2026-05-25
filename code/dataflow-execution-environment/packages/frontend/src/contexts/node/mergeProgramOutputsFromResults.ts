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
    current.denominator === next.denominator
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
        return {
          ...n,
          data: {
            ...opData,
            ...newData,
            result:
              newData.value ??
              (resultValue.kind === "semantic"
                ? resultValue.result.totalAmount
                : undefined),
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
