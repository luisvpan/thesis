import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import { useEdgesState, useNodesState, type Edge } from "@xyflow/react";
import { useVision } from "./VisionContext";
import { getNodeValue, getRightmostEvaluableNode } from "./node/helpers";
import type { DataflowNode, NodeContextState, PortIdentifier, PortKindInfo, ShakingPort } from "./node/types";
import { useFlowGraphEffects } from "./node/useFlowGraphEffects";
import { useManualExecuteProgram } from "./node/useManualExecuteProgram";
import { useNodeSpawning } from "./node/useNodeSpawning";
import { usePortSelection } from "./node/usePortSelection";
import { useProgramExecutorRef } from "./node/useProgramExecutorRef";
import { computeNodeIdsInsideActiveArrayZones } from "@/utils/arrayZoneGeometry";

export type { DataflowNode, PortDefinition, PortIdentifier, PortKindInfo } from "./node/types";

const NodeContext = createContext<NodeContextState | null>(null);

type NodeProviderProps = {
  children: ReactNode;
  flowContainerRef: RefObject<HTMLDivElement | null>;
  visionSyncEnabled?: boolean;
  nodesDraggable?: boolean;
};

export function NodeProvider({
  children,
  flowContainerRef,
  visionSyncEnabled = true,
  nodesDraggable = false,
}: NodeProviderProps) {
  const { lastCardFrame } = useVision();
  const executorRef = useProgramExecutorRef();

  const [nodes, setNodes, onNodesChange] = useNodesState<DataflowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedPort, setSelectedPort] = useState<PortIdentifier | null>(null);

  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<number | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [shakingPort, setShakingPort] = useState<ShakingPort>(null);

  // Port kind registry: Map<"nodeId:handleId", PortKindInfo>
  const portKindRegistryRef = useRef<Map<string, PortKindInfo>>(new Map());

  const registerPortKind = useCallback((nodeId: string, handleId: string, info: PortKindInfo) => {
    const key = `${nodeId}:${handleId}`;
    portKindRegistryRef.current.set(key, info);
  }, []);

  const unregisterPortKinds = useCallback((nodeId: string) => {
    const keysToDelete: string[] = [];
    for (const key of portKindRegistryRef.current.keys()) {
      if (key.startsWith(`${nodeId}:`)) {
        keysToDelete.push(key);
      }
    }
    for (const key of keysToDelete) {
      portKindRegistryRef.current.delete(key);
    }
  }, []);

  const getPortKindInfo = useCallback((nodeId: string, handleId: string): PortKindInfo | undefined => {
    const key = `${nodeId}:${handleId}`;
    return portKindRegistryRef.current.get(key);
  }, []);

  const triggerIncompatibleFeedback = useCallback((nodeId: string, handleId: string) => {
    setShakingPort({ nodeId, handleId });
    setTimeout(() => setShakingPort(null), 300);
  }, []);

  useFlowGraphEffects({
    visionSyncEnabled,
    lastCardFrame,
    flowContainerRef,
    setNodes,
    nodes,
    edges,
    setEdges,
    executorRef,
    setExecutionError,
    setExecutionResult,
  });

  const { isPortSelected, clearSelection, handlePortClick } = usePortSelection(
    selectedPort,
    setSelectedPort,
    setEdges,
    nodes,
    getPortKindInfo,
    triggerIncompatibleFeedback
  );

  const {
    getNodePorts,
    addNumberNode,
    addOperatorNode,
    addResultAnchorPair,
    addResultCard,
    spawnDeckYoloClass,
    addArrayOpenNode,
    addArrayCloseNode,
  } = useNodeSpawning(setNodes);

  const executeProgram = useManualExecuteProgram(
    executorRef,
    nodes,
    edges,
    setNodes,
    setIsExecuting,
    setExecutionError,
    setExecutionResult
  );

  const getExecutionResult = useCallback(() => {
    const rightmost = getRightmostEvaluableNode(nodes);
    const value = rightmost ? getNodeValue(rightmost) : undefined;
    return typeof value === "number" ? value : null;
  }, [nodes]);

  const nodesInsideArrayZones = useMemo(
    () => computeNodeIdsInsideActiveArrayZones(nodes, edges),
    [nodes, edges]
  );

  const isNodeInsideArrayZone = useCallback(
    (nodeId: string) => nodesInsideArrayZones.has(nodeId),
    [nodesInsideArrayZones]
  );

  const value = useMemo(
    (): NodeContextState => ({
      nodes,
      edges,
      selectedPort,
      isExecuting,
      executionResult,
      executionError,
      getNodePorts,
      isPortSelected,
      isNodeInsideArrayZone,
      handlePortClick,
      clearSelection,
      addNumberNode,
      addOperatorNode,
      addResultAnchorPair,
      addResultCard,
      spawnDeckYoloClass,
      addArrayOpenNode,
      addArrayCloseNode,
      nodesDraggable,
      executeProgram,
      onNodesChange,
      onEdgesChange,
      getExecutionResult,
      registerPortKind,
      unregisterPortKinds,
      getPortKindInfo,
      shakingPort,
    }),
    [
      nodes,
      edges,
      selectedPort,
      isExecuting,
      executionResult,
      executionError,
      getNodePorts,
      isPortSelected,
      isNodeInsideArrayZone,
      handlePortClick,
      clearSelection,
      addNumberNode,
      addOperatorNode,
      addResultAnchorPair,
      addResultCard,
      spawnDeckYoloClass,
      addArrayOpenNode,
      addArrayCloseNode,
      nodesDraggable,
      executeProgram,
      onNodesChange,
      onEdgesChange,
      getExecutionResult,
      registerPortKind,
      unregisterPortKinds,
      getPortKindInfo,
      shakingPort,
    ]
  );

  return <NodeContext.Provider value={value}>{children}</NodeContext.Provider>;
}

export function useNode(): NodeContextState {
  const ctx = useContext(NodeContext);
  if (!ctx) {
    throw new Error("useNode must be used within NodeProvider");
  }
  return ctx;
}
