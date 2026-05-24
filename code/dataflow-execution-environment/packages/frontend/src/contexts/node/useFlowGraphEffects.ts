import {
  useEffect,
  useRef,
  type Dispatch,
  type MutableRefObject,
  type RefObject,
  type SetStateAction,
} from "react";
import type { Edge } from "@xyflow/react";
import type { ProgramExecutor } from "@/services/executeProgram";
import { computeProgramHash } from "@/services/executeProgram";
import type { CardDetectionsPayload } from "../VisionContext";
import { VISION_FLOW_MIN_SIZE } from "./constants";
import { mergeProgramOutputsFromResults } from "./mergeProgramOutputsFromResults";
import { mergeVisionFrameIntoNodes } from "./mergeVisionFrameIntoNodes";
import { applyNumberTouchMerge } from "@/utils/numberTouchMerge";
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
}: UseFlowGraphEffectsParams): void {
  const lastProgramHashRef = useRef<string | null>(null);

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

  useEffect(() => {
    const evalNodes = nodes.filter(
      (n) => n.type === "source" || n.type === "operator"
    );

    if (evalNodes.length === 0 || !executorRef.current) return;

    const programHash = computeProgramHash(nodes, edges);
    if (programHash === lastProgramHashRef.current) return;
    lastProgramHashRef.current = programHash;

    executorRef.current.execute(nodes, edges).then((result) => {
      if (result.success && result.results) {
        setExecutionError(null);

        setNodes((nds) => {
          const merged = mergeProgramOutputsFromResults(nds, result.results!);
          return merged === nds ? nds : merged;
        });
      } else if (result.error) {
        setExecutionResult(null);
        setExecutionError(result.error);
      }
    });
  }, [nodes, edges, setNodes, executorRef, setExecutionError, setExecutionResult]);
}
