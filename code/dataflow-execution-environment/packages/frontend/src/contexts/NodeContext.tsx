import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import { useEdgesState, useNodesState, type Edge } from "@xyflow/react";
import { useVision } from "./VisionContext";
import { getNodeValue, getRightmostEvaluableNode } from "./node/helpers";
import type { DataflowNode, NodeContextState, PortIdentifier } from "./node/types";
import { useFlowGraphEffects } from "./node/useFlowGraphEffects";
import { useManualExecuteProgram } from "./node/useManualExecuteProgram";
import { useNodeSpawning } from "./node/useNodeSpawning";
import { usePortSelection } from "./node/usePortSelection";
import { useProgramExecutorRef } from "./node/useProgramExecutorRef";

export type { DataflowNode, PortDefinition, PortIdentifier } from "./node/types";

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
    setEdges
  );

  const {
    getNodePorts,
    addNumberNode,
    addOperatorNode,
    addResultAnchorPair,
    addResultCard,
    spawnDeckYoloClass,
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
      handlePortClick,
      clearSelection,
      addNumberNode,
      addOperatorNode,
      addResultAnchorPair,
      addResultCard,
      spawnDeckYoloClass,
      nodesDraggable,
      executeProgram,
      onNodesChange,
      onEdgesChange,
      getExecutionResult,
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
      handlePortClick,
      clearSelection,
      addNumberNode,
      addOperatorNode,
      addResultAnchorPair,
      addResultCard,
      spawnDeckYoloClass,
      nodesDraggable,
      executeProgram,
      onNodesChange,
      onEdgesChange,
      getExecutionResult,
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
