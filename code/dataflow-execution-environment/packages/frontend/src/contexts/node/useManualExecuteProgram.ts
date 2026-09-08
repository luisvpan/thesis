import { useCallback, type Dispatch, type MutableRefObject, type SetStateAction } from "react";
import type { Edge } from "@xyflow/react";
import type { ProgramExecutor, ResultValue } from "@/services/executeProgram";
import { mergeProgramOutputsFromResults } from "./mergeProgramOutputsFromResults";
import type { DataflowNode } from "./types";
import { logger } from "@/lib/logger";

type SetNodes = Dispatch<SetStateAction<DataflowNode[]>>;

export function useManualExecuteProgram(
  executorRef: MutableRefObject<ProgramExecutor | null>,
  nodes: DataflowNode[],
  edges: Edge[],
  setNodes: SetNodes,
  setIsExecuting: (v: boolean) => void,
  setExecutionError: (msg: string | null) => void,
  setExecutionResult: (n: number | null) => void,
  setEvalResults: (results: Map<string, ResultValue>) => void
) {
  return useCallback(async () => {
    if (!executorRef.current) {
      logger.executeProgram.error("No executor available");
      return;
    }

    setIsExecuting(true);
    setExecutionError(null);

    try {
      const result = await executorRef.current.execute(nodes, edges);

      if (result.success && result.results && result.results.size > 0) {
        setEvalResults(new Map(result.results));
        setNodes((nds) => mergeProgramOutputsFromResults(nds, result.results!));

        const firstResult = result.results.values().next().value;
        const numericResult =
          firstResult?.kind === "number"
            ? firstResult.value
            : firstResult?.result.totalAmount;
        setExecutionResult(numericResult ?? null);
        setExecutionError(null);

        const stats = executorRef.current.getStats();
        logger.executeProgram.debug("Execution stats", { stats });
      } else {
        setExecutionResult(null);
        setEvalResults(new Map());
        setExecutionError(result.error || "Error desconocido");
      }
    } catch (err) {
      setExecutionResult(null);
      setEvalResults(new Map());
      setExecutionError(err instanceof Error ? err.message : "Error de ejecución");
    } finally {
      setIsExecuting(false);
    }
  }, [
    nodes,
    edges,
    setNodes,
    executorRef,
    setIsExecuting,
    setExecutionError,
    setExecutionResult,
    setEvalResults,
  ]);
}
