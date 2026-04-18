import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  type OnNodesChange,
  type OnEdgesChange,
} from "@xyflow/react";
import { useVision } from "./VisionContext";
import { visionLabelToDigit } from "@/utils/visionCardLabel";
import type { NumberFlowNodeData, OperatorFlowNodeData } from "@/components/dataflow";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type DataflowNode =
  | Node<NumberFlowNodeData, "number">
  | Node<OperatorFlowNodeData, "operator">;

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

type NodeContextState = {
  // Estado
  nodes: DataflowNode[];
  edges: Edge[];
  selectedPort: PortIdentifier | null;

  // Consultas
  getNodePorts: (nodeType: "number" | "operator") => PortDefinition[];
  isPortSelected: (
    nodeId: string,
    handleId: string,
    handleType: "source" | "target"
  ) => boolean;

  // Acciones
  handlePortClick: (
    nodeId: string,
    handleId: string,
    handleType: "source" | "target"
  ) => void;
  clearSelection: () => void;
  addNumberNode: (value: number, position?: { x: number; y: number }) => void;
  addOperatorNode: (
    operator: "adicion" | "sustraccion",
    position?: { x: number; y: number }
  ) => void;

  // Para React Flow
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;

  // Resultado calculado
  getExecutionResult: () => number | null;
};

const NodeContext = createContext<NodeContextState | null>(null);

// ─────────────────────────────────────────────────────────────────────────────
// Port Definitions
// ─────────────────────────────────────────────────────────────────────────────

const NUMBER_PORTS: PortDefinition[] = [
  { handleId: "in", handleType: "target", position: "left" },
  { handleId: "out", handleType: "source", position: "right" },
];

const OPERATOR_PORTS: PortDefinition[] = [
  { handleId: "a", handleType: "target", position: "left", offsetY: "25%" },
  { handleId: "b", handleType: "target", position: "left", offsetY: "75%" },
  { handleId: "out", handleType: "source", position: "right" },
];

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

const VISION_FLOW_MIN_SIZE = 64;
const VISION_NODE_HALF_W = 48;
const VISION_NODE_HALF_H = 40;

/**
 * Convierte coordenadas normalizadas (0-1) a coordenadas del ReactFlow.
 * El CV system envía posiciones normalizadas respecto al viewport del proyector.
 * Se multiplica por el tamaño del viewport y se resta el offset del contenedor.
 */
function visionToFlowPosition(
  pos: { x: number; y: number },
  flowRect: Pick<DOMRectReadOnly, "left" | "top" | "width" | "height">
): { x: number; y: number } {
  if (
    flowRect.width < VISION_FLOW_MIN_SIZE ||
    flowRect.height < VISION_FLOW_MIN_SIZE
  ) {
    return { x: 0, y: 0 };
  }

  // Convertir de coordenadas normalizadas (0-1) a píxeles del viewport
  const viewportX = pos.x * window.innerWidth;
  const viewportY = pos.y * window.innerHeight;

  // Restar el offset del contenedor para obtener coordenadas locales al ReactFlow
  let x = viewportX - flowRect.left - VISION_NODE_HALF_W;
  let y = viewportY - flowRect.top - VISION_NODE_HALF_H;

  // Clamp a los límites del contenedor
  const maxX = Math.max(0, flowRect.width - 2 * VISION_NODE_HALF_W);
  const maxY = Math.max(0, flowRect.height - 2 * VISION_NODE_HALF_H);
  x = Math.max(0, Math.min(x, maxX));
  y = Math.max(0, Math.min(y, maxY));

  return { x, y };
}

function getNodeValue(node: DataflowNode | null | undefined): number | undefined {
  if (!node?.data) return undefined;
  const d = node.data as NumberFlowNodeData & OperatorFlowNodeData;
  return d.value ?? d.result;
}

function getRightmostNode(nodes: DataflowNode[]): DataflowNode | null {
  if (nodes.length === 0) return null;
  return nodes.reduce((rightmost, node) =>
    node.position.x > rightmost.position.x ? node : rightmost
  );
}

function computeOperatorResult(
  operatorId: string,
  nodes: DataflowNode[],
  edges: Edge[]
): number | undefined {
  const edgesToOperator = edges.filter((e) => e.target === operatorId);
  const edgeA = edgesToOperator.find((e) => e.targetHandle === "a");
  const edgeB = edgesToOperator.find((e) => e.targetHandle === "b");
  const nodeA = edgeA ? nodes.find((n) => n.id === edgeA.source) : null;
  const nodeB = edgeB ? nodes.find((n) => n.id === edgeB.source) : null;
  const valA = getNodeValue(nodeA);
  const valB = getNodeValue(nodeB);
  if (typeof valA !== "number" || typeof valB !== "number") return undefined;
  const operator = (
    nodes.find((n) => n.id === operatorId)?.data as OperatorFlowNodeData | undefined
  )?.operator;
  if (operator === "adicion") return valA + valB;
  if (operator === "sustraccion") return valA - valB;
  return undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────

type NodeProviderProps = {
  children: ReactNode;
  flowContainerRef: React.RefObject<HTMLDivElement | null>;
};

export function NodeProvider({ children, flowContainerRef }: NodeProviderProps) {
  const { lastCardFrame } = useVision();

  const [nodes, setNodes, onNodesChange] = useNodesState<DataflowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedPort, setSelectedPort] = useState<PortIdentifier | null>(null);

  // Sync vision cards to nodes
  useEffect(() => {
    if (!lastCardFrame) return;
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
      const withoutLive = prev.filter((n) => !n.id.startsWith("vision-live-"));
      const additions: DataflowNode[] = lastCardFrame.cards.map((c, i) => {
        const digit = visionLabelToDigit(c.label);
        // c.position viene en coordenadas absolutas del viewport (píxeles)
        const position = visionToFlowPosition(c.position, rect);
        return {
          id: `vision-live-${i}`,
          type: "number" as const,
          position,
          data: {
            value: digit ?? 0,
            visionSubtitle: digit == null ? c.label : undefined,
          },
        };
      });
      return [...withoutLive, ...additions];
    });
  }, [lastCardFrame, setNodes, flowContainerRef]);

  // Compute operator results when edges change
  useEffect(() => {
    setNodes((nds) => {
      let current = nds;
      for (let pass = 0; pass < 10; pass++) {
        let changed = false;
        current = current.map((node) => {
          if (node.type !== "operator") return node;
          const result = computeOperatorResult(node.id, current, edges);
          const prev = (node.data as OperatorFlowNodeData).result;
          if (result !== prev) changed = true;
          return {
            ...node,
            data: { ...node.data, result, value: result },
          };
        });
        if (!changed) break;
      }
      return current;
    });
  }, [edges, setNodes]);

  // Port selection logic
  const isPortSelected = useCallback(
    (nodeId: string, handleId: string, handleType: "source" | "target") => {
      return (
        selectedPort?.nodeId === nodeId &&
        selectedPort?.handleId === handleId &&
        selectedPort?.handleType === handleType
      );
    },
    [selectedPort]
  );

  const clearSelection = useCallback(() => {
    setSelectedPort(null);
  }, []);

  const handlePortClick = useCallback(
    (nodeId: string, handleId: string, handleType: "source" | "target") => {
      if (!selectedPort) {
        setSelectedPort({ nodeId, handleId, handleType });
        return;
      }

      // Same port clicked: deselect
      if (
        selectedPort.nodeId === nodeId &&
        selectedPort.handleId === handleId &&
        selectedPort.handleType === handleType
      ) {
        setSelectedPort(null);
        return;
      }

      const first = selectedPort;
      const second = { nodeId, handleId, handleType };

      // Both same type: replace selection
      if (first.handleType === second.handleType) {
        setSelectedPort(second);
        return;
      }

      // Self-connection: replace selection
      if (first.nodeId === second.nodeId) {
        setSelectedPort(second);
        return;
      }

      // Valid connection
      const source = first.handleType === "source" ? first : second;
      const target = first.handleType === "target" ? first : second;

      const connection: Connection = {
        source: source.nodeId,
        sourceHandle: source.handleId,
        target: target.nodeId,
        targetHandle: target.handleId,
      };

      setEdges((eds) => addEdge(connection, eds));
      setSelectedPort(null);
    },
    [selectedPort, setEdges]
  );

  const getNodePorts = useCallback((nodeType: "number" | "operator") => {
    return nodeType === "number" ? NUMBER_PORTS : OPERATOR_PORTS;
  }, []);

  const addNumberNode = useCallback(
    (value: number, position?: { x: number; y: number }) => {
      const id = `num-${value}-${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "number" as const,
          position: position ?? {
            x: 100 + (nds.length % 3) * 60,
            y: 80 + Math.floor(nds.length / 3) * 100,
          },
          data: { value },
        },
      ]);
    },
    [setNodes]
  );

  const addOperatorNode = useCallback(
    (operator: "adicion" | "sustraccion", position?: { x: number; y: number }) => {
      const id = `op-${operator}-${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "operator" as const,
          position: position ?? {
            x: 320 + (nds.filter((n) => n.type === "operator").length % 2) * 200,
            y: 120,
          },
          data: { operator },
        },
      ]);
    },
    [setNodes]
  );

  const getExecutionResult = useCallback(() => {
    const rightmost = getRightmostNode(nodes);
    const value = rightmost ? getNodeValue(rightmost) : undefined;
    return typeof value === "number" ? value : null;
  }, [nodes]);

  const value = useMemo(
    (): NodeContextState => ({
      nodes,
      edges,
      selectedPort,
      getNodePorts,
      isPortSelected,
      handlePortClick,
      clearSelection,
      addNumberNode,
      addOperatorNode,
      onNodesChange,
      onEdgesChange,
      getExecutionResult,
    }),
    [
      nodes,
      edges,
      selectedPort,
      getNodePorts,
      isPortSelected,
      handlePortClick,
      clearSelection,
      addNumberNode,
      addOperatorNode,
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
