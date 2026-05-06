import type { Edge, OnEdgesChange, OnNodesChange } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import type {
  OperatorFlowNodeData,
  ProgramOutputFlowNodeData,
  SourceFlowNodeData,
  ArrayOpenNodeData,
  ArrayCloseNodeData,
} from "@/components/dataflow";
import type { OperatorType } from "@/types/card-types";

export type DataflowNode =
  | Node<SourceFlowNodeData, "source">
  | Node<OperatorFlowNodeData, "operator">
  | Node<ProgramOutputFlowNodeData, "programOutput">
  | Node<ArrayOpenNodeData, "arrayOpen">
  | Node<ArrayCloseNodeData, "arrayClose">;

export type PortIdentifier = {
  nodeId: string;
  handleId: string;
  handleType: "source" | "target";
};

export type PortDefinition = {
  handleId: string;
  handleType: "source" | "target";
  position: "left" | "right";
  offsetY?: string;
};

export type NodeContextState = {
  nodes: DataflowNode[];
  edges: Edge[];
  selectedPort: PortIdentifier | null;

  isExecuting: boolean;
  executionResult: number | null;
  executionError: string | null;

  getNodePorts: (nodeType: "source" | "operator") => PortDefinition[];
  isPortSelected: (
    nodeId: string,
    handleId: string,
    handleType: "source" | "target"
  ) => boolean;

  handlePortClick: (
    nodeId: string,
    handleId: string,
    handleType: "source" | "target"
  ) => void;
  clearSelection: () => void;
  addNumberNode: (value: number, position?: { x: number; y: number }) => void;
  addOperatorNode: (
    operator: OperatorType,
    position?: { x: number; y: number }
  ) => void;
  addResultAnchorPair: () => void;
  addResultCard: () => void;
  spawnDeckYoloClass: (yoloClass: string) => void;
  addArrayOpenNode: () => void;
  addArrayCloseNode: () => void;
  nodesDraggable: boolean;
  executeProgram: () => Promise<void>;

  onNodesChange: OnNodesChange<DataflowNode>;
  onEdgesChange: OnEdgesChange;

  getExecutionResult: () => number | null;
};
