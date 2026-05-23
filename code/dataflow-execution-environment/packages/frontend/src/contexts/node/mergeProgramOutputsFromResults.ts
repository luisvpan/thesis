import type { ProgramOutputFlowNodeData } from "@/components/dataflow";
import type { ResultValue } from "@/services/executeProgram";
import type { DataflowNode } from "./types";

/** Aplica resultados del intérprete a nodos `programOutput` sin tocar posiciones ni otros tipos. */
export function mergeProgramOutputsFromResults(
  nodes: DataflowNode[],
  results: Map<string, ResultValue>
): DataflowNode[] {
  let changed = false;
  const updated = nodes.map((n) => {
    if (n.type !== "programOutput") return n;
    const resultValue = results.get(n.id);
    if (resultValue === undefined) return n;

    const currentData = n.data as ProgramOutputFlowNodeData;
    let newData: ProgramOutputFlowNodeData;

    if (resultValue.kind === "number") {
      newData = { value: resultValue.value, description: undefined, visualStrip: undefined };
    } else {
      newData = {
        value: resultValue.result.totalAmount,
        description: resultValue.result.description,
        visualStrip: resultValue.result.visualStrip,
        originalElements: resultValue.result.originalElements,
        isSingleCpaObject: resultValue.isSingleCpaObject,
        singleCpaObjectMeta: resultValue.singleCpaObjectMeta,
      };
    }

    if (
      currentData.value === newData.value &&
      currentData.description === newData.description &&
      JSON.stringify(currentData.visualStrip) === JSON.stringify(newData.visualStrip) &&
      currentData.isSingleCpaObject === newData.isSingleCpaObject &&
      JSON.stringify(currentData.singleCpaObjectMeta) === JSON.stringify(newData.singleCpaObjectMeta)
    ) {
      return n;
    }

    changed = true;
    return { ...n, data: { ...n.data, ...newData } };
  });
  return changed ? updated : nodes;
}
