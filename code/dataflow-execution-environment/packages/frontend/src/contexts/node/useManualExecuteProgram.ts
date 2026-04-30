import { useCallback, type Dispatch, type MutableRefObject, type SetStateAction } from "react";
import type { Edge } from "@xyflow/react";
import type { ProgramOutputFlowNodeData } from "@/components/dataflow";
import type { ProgramExecutor } from "@/services/executeProgram";
import type { DataflowNode } from "./types";

type SetNodes = Dispatch<SetStateAction<DataflowNode[]>>;

export function useManualExecuteProgram(
  executorRef: MutableRefObject<ProgramExecutor | null>,
  nodes: DataflowNode[],
  edges: Edge[],
  setNodes: SetNodes,
  setIsExecuting: (v: boolean) => void,
  setExecutionError: (msg: string | null) => void,
  setExecutionResult: (n: number | null) => void
) {
  return useCallback(async () => {
    if (!executorRef.current) {
      console.error("[executeProgram] No executor available");
      return;
    }

    setIsExecuting(true);
    setExecutionError(null);

    const syncProgramOutputValue = (num: number | undefined) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.type === "programOutput"
            ? {
                ...n,
                data: {
                  ...(n.data as ProgramOutputFlowNodeData),
                  value: num,
                },
              }
            : n
        )
      );
    };

    try {
      const result = await executorRef.current.execute(nodes, edges);

      if (result.success && result.results && result.results.size > 0) {
        const firstResult = result.results.values().next().value;
        const numericResult =
          firstResult?.kind === "number"
            ? firstResult.value
            : firstResult?.result.totalAmount;
        setExecutionResult(numericResult ?? null);
        setExecutionError(null);
        syncProgramOutputValue(typeof numericResult === "number" ? numericResult : undefined);

        const stats = executorRef.current.getStats();
        console.log("[executeProgram] Stats:", stats);
      } else {
        setExecutionResult(null);
        setExecutionError(result.error || "Error desconocido");
        syncProgramOutputValue(undefined);
      }
    } catch (err) {
      setExecutionResult(null);
      setExecutionError(err instanceof Error ? err.message : "Error de ejecución");
      syncProgramOutputValue(undefined);
    } finally {
      setIsExecuting(false);
    }
  }, [nodes, edges, setNodes, executorRef, setIsExecuting, setExecutionError, setExecutionResult]);
}
