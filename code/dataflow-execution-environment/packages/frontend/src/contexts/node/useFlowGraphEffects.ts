import {
  useEffect,
  type Dispatch,
  type MutableRefObject,
  type RefObject,
  type SetStateAction,
} from "react";
import type { Edge } from "@xyflow/react";
import type { ProgramExecutor, ResultValue } from "@/services/executeProgram";
import type { CardDetectionsPayload } from "../VisionContext";
import { VISION_FLOW_MIN_SIZE } from "./constants";
import { mergeProgramOutputsFromResults } from "./mergeProgramOutputsFromResults";
import { applyErrorMarks } from "./errorMarks";
import { mergeVisionFrameIntoNodes } from "./mergeVisionFrameIntoNodes";
import { applyNumberTouchMerge } from "@/utils/numberTouchMerge";
import { logger } from "@/lib/logger";
import type { DataflowNode } from "./types";

type SetNodes = Dispatch<SetStateAction<DataflowNode[]>>;
type SetEdges = Dispatch<SetStateAction<Edge[]>>;

type UseFlowGraphEffectsParams = {
  visionSyncEnabled: boolean;
  nodesDraggable: boolean;
  lastCardFrame: CardDetectionsPayload | null;
  flowContainerRef: RefObject<HTMLDivElement | null>;
  setNodes: SetNodes;
  nodes: DataflowNode[];
  edges: Edge[];
  setEdges: SetEdges;
  executorRef: MutableRefObject<ProgramExecutor | null>;
  setExecutionError: (msg: string | null) => void;
  setExecutionResult: (n: number | null) => void;
  setEvalResults: (results: Map<string, ResultValue>) => void;
};

export function useFlowGraphEffects({
  visionSyncEnabled,
  nodesDraggable,
  lastCardFrame,
  flowContainerRef,
  setNodes,
  nodes,
  edges,
  setEdges,
  executorRef,
  setExecutionError,
  setExecutionResult,
  setEvalResults,
}: UseFlowGraphEffectsParams): void {
  useEffect(() => {
    if (!visionSyncEnabled || !lastCardFrame) return;
    const flowEl = flowContainerRef.current;
    const rect = flowEl?.getBoundingClientRect();
    if (
      !rect ||
      rect.width < VISION_FLOW_MIN_SIZE ||
      rect.height < VISION_FLOW_MIN_SIZE
    ) {
      return;
    }

    setNodes((prev) => {
      const next = applyNumberTouchMerge(
        mergeVisionFrameIntoNodes(prev, lastCardFrame, rect, nodesDraggable)
      );
      return next === prev ? prev : next;
    });
  }, [
    visionSyncEnabled,
    nodesDraggable,
    lastCardFrame,
    setNodes,
    flowContainerRef,
  ]);

  useEffect(() => {
    const nodeIds = new Set(nodes.map((n) => n.id));

    setEdges((eds) => {
      const validEdges = eds.filter(
        (e) => nodeIds.has(e.source) && nodeIds.has(e.target)
      );
      return validEdges.length === eds.length ? eds : validEdges;
    });
  }, [nodes, setEdges]);

  // Remove diceZone nodes that are disconnected AND have no active dice card
  useEffect(() => {
    setNodes((nds) => {
      const toRemove = nds.filter((n) => {
        if (n.type !== "diceZone") return false;
        const hasEdge = edges.some((e) => e.source === n.id);
        if (hasEdge) return false;
        const status = (n.data as { visionStatus?: string }).visionStatus;
        return status !== "active";
      });
      if (toRemove.length === 0) return nds;
      const removeIds = new Set(toRemove.map((n) => n.id));
      return nds.filter((n) => !removeIds.has(n.id));
    });
  }, [nodes, edges, setNodes]);

  useEffect(() => {
    const evalNodes = nodes.filter(
      (n) => n.type === "source" || n.type === "operator"
    );

    if (evalNodes.length === 0 || !executorRef.current) return;

    // Sin compuerta aquí: el ejecutor corta solo si el programa no cambió, que
    // es quien lo tiene escrito delante. Un lienzo quieto devuelve el mismo
    // resultado y las guardas de identidad de abajo evitan el render.
    executorRef.current
      .execute(nodes, edges)
      .then((result) => {
        // Una salida rota no borra el valor de las demás: se aplican los
        // resultados que sí hay y cada error va a su carta.
        setExecutionError(result.programError);
        setEvalResults(new Map(result.results));

        setNodes((nds) => {
          const merged = applyErrorMarks(
            mergeProgramOutputsFromResults(nds, result.results, result.errorsByOutput),
            edges,
            result.errorsByOutput
          );
          return merged === nds ? nds : merged;
        });

        if (result.programError) setExecutionResult(null);
      })
      .catch((err) => {
        logger.execute.error("Unhandled execution error", {
          error: err instanceof Error ? err.message : String(err),
        });
        setExecutionResult(null);
        setExecutionError(
          err instanceof Error ? err.message : "Error de ejecución"
        );
      });
  }, [nodes, edges, setNodes, executorRef, setExecutionError, setExecutionResult, setEvalResults]);
}
