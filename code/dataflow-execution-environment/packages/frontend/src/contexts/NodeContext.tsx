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
import { parseVisionLabel, visionOperatorToMathOperator } from "@/types/vision-card";
import { executeProgram as executeProgramService } from "@/services/executeProgram";
import type {
  NumberFlowNodeData,
  OperatorFlowNodeData,
  ResultAnchorFlowNodeData,
  ProgramOutputFlowNodeData,
} from "@/components/dataflow";
import type { MathOperatorType } from "@/types/card-types";
import {
  VISION_PROGRAM_OUTPUT_ID,
  VISION_RESULT_ANCHOR_ID,
} from "@/utils/frontendFlowConstants";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type DataflowNode =
  | Node<NumberFlowNodeData, "number">
  | Node<OperatorFlowNodeData, "operator">
  | Node<ResultAnchorFlowNodeData, "resultAnchor">
  | Node<ProgramOutputFlowNodeData, "programOutput">;

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

  // Estado de ejecución
  isExecuting: boolean;
  executionResult: number | null;
  executionError: string | null;

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
    operator: MathOperatorType,
    position?: { x: number; y: number }
  ) => void;
  executeProgram: () => Promise<void>;

  // Para React Flow
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;

  // Resultado calculado (local, sin backend)
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

/** Coincide con `w-60` del lienzo + margen hasta la carta de resultado */
const VISION_CARD_BOX = 240;
const VISION_RESULT_GAP = 24;

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
  if (node.type === "programOutput") {
    const v = (node.data as ProgramOutputFlowNodeData).value;
    return typeof v === "number" ? v : undefined;
  }
  if (node.type === "resultAnchor") return undefined;
  const d = node.data as NumberFlowNodeData & OperatorFlowNodeData;
  return d.value ?? d.result;
}

function getRightmostEvaluableNode(nodes: DataflowNode[]): DataflowNode | null {
  const evalNodes = nodes.filter(
    (n): n is Extract<DataflowNode, { type: "number" | "operator" }> =>
      n.type === "number" || n.type === "operator"
  );
  if (evalNodes.length === 0) return null;
  return evalNodes.reduce((rightmost, node) =>
    node.position.x > rightmost.position.x ? node : rightmost
  );
}

/**
 * Genera un slug válido para el DSL (solo letras y _) a partir del trackId.
 * Si no hay trackId válido, usa el índice como fallback.
 */
function toValidSlug(trackId: number | undefined, fallbackIndex: number): string {
  if (trackId !== undefined && trackId >= 0) {
    return `card_${trackId}`;
  }
  return `card_${fallbackIndex}`;
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
  switch (operator) {
    case "adicion":
      return valA + valB;
    case "sustraccion":
      return valA - valB;
    case "multiplicacion":
      return valA * valB;
    case "division":
      return valB !== 0 ? valA / valB : undefined;
    default:
      return undefined;
  }
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

  // Estado de ejecución
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<number | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);

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
      const prevOut = prev.find(
        (n) => n.id === VISION_PROGRAM_OUTPUT_ID && n.type === "programOutput"
      );
      const preservedValue =
        prevOut?.type === "programOutput"
          ? (prevOut.data as ProgramOutputFlowNodeData).value
          : undefined;

      const withoutLive = prev.filter((n) => !n.id.startsWith("card_"));

      let grapesFlowPos: { x: number; y: number } | null = null;
      const additions: DataflowNode[] = [];
      let idx = 0;

      for (const c of lastCardFrame.cards) {
        const parsed = parseVisionLabel(c.label);

        if (parsed.type === "resultAnchor") {
          grapesFlowPos = visionToFlowPosition(c.position, rect);
          continue;
        }

        const position = visionToFlowPosition(c.position, rect);
        const nodeId = toValidSlug(c.trackId, idx);

        if (parsed.type === "number") {
          additions.push({
            id: nodeId,
            type: "number" as const,
            position,
            data: {
              value: parsed.value,
              trackId: c.trackId,
            },
          });
          continue;
        }

        if (parsed.type === "operator") {
          additions.push({
            id: nodeId,
            type: "operator" as const,
            position,
            data: {
              operator: visionOperatorToMathOperator(parsed.operator),
              trackId: c.trackId,
            },
          });
          continue;
        }

        // Tipo desconocido: mostrar como número con subtítulo
        additions.push({
          id: nodeId,
          type: "number" as const,
          position,
          data: {
            value: 0,
            visionSubtitle: parsed.label,
            trackId: c.trackId,
          },
        });
      }

      if (grapesFlowPos) {
        additions.push(
          {
            id: VISION_RESULT_ANCHOR_ID,
            type: "resultAnchor" as const,
            position: grapesFlowPos,
            data: {},
          },
          {
            id: VISION_PROGRAM_OUTPUT_ID,
            type: "programOutput" as const,
            position: {
              x: grapesFlowPos.x + VISION_CARD_BOX + VISION_RESULT_GAP,
              y: grapesFlowPos.y,
            },
            data: { value: preservedValue },
          }
        );
      }

      return [...withoutLive, ...additions];
    });
  }, [lastCardFrame, setNodes, flowContainerRef]);

  // Arista por defecto: marcador → carta de resultado (solo presentación)
  useEffect(() => {
    const hasAnchor = nodes.some(
      (n) => n.id === VISION_RESULT_ANCHOR_ID && n.type === "resultAnchor"
    );
    const hasOutput = nodes.some(
      (n) => n.id === VISION_PROGRAM_OUTPUT_ID && n.type === "programOutput"
    );
    if (!hasAnchor || !hasOutput) return;

    setEdges((eds) => {
      const exists = eds.some(
        (e) =>
          e.source === VISION_RESULT_ANCHOR_ID &&
          e.sourceHandle === "out" &&
          e.target === VISION_PROGRAM_OUTPUT_ID &&
          e.targetHandle === "in"
      );
      if (exists) return eds;
      return addEdge(
        {
          source: VISION_RESULT_ANCHOR_ID,
          sourceHandle: "out",
          target: VISION_PROGRAM_OUTPUT_ID,
          targetHandle: "in",
        },
        eds
      );
    });
  }, [nodes, setEdges]);

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
      const id = `num${value}_${Date.now()}`;
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
    (operator: MathOperatorType, position?: { x: number; y: number }) => {
      const id = `op${operator}_${Date.now()}`;
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

  const executeProgram = useCallback(async () => {
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
      const result = await executeProgramService(nodes, edges);

      if (result.success && result.result !== undefined) {
        setExecutionResult(result.result);
        setExecutionError(null);
        syncProgramOutputValue(result.result);
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
  }, [nodes, edges, setNodes]);

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
