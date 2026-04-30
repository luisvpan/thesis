import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useMemo,
  useRef,
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
import { createProgramExecutor, type ProgramExecutor, type ResultValue } from "@/services/executeProgram";
import type {
  SourceFlowNodeData,
  OperatorFlowNodeData,
  ProgramOutputFlowNodeData,
} from "@/components/dataflow";
import type { OperatorType } from "@/types/card-types";
import { spawnActionForYoloClass } from "../data/yoloDeckCatalog";
import { VISION_PROGRAM_OUTPUT_ID } from "@/utils/frontendFlowConstants";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type DataflowNode =
  | Node<SourceFlowNodeData, "source">
  | Node<OperatorFlowNodeData, "operator">
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
  getNodePorts: (nodeType: "source" | "operator") => PortDefinition[];
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
    operator: OperatorType,
    position?: { x: number; y: number }
  ) => void;
  addResultAnchorPair: () => void;
  addResultCard: () => void;
  spawnDeckYoloClass: (yoloClass: string) => void;
  nodesDraggable: boolean;
  executeProgram: () => Promise<void>;

  // Para React Flow
  onNodesChange: OnNodesChange<DataflowNode>;
  onEdgesChange: OnEdgesChange;

  // Resultado calculado (local, sin backend)
  getExecutionResult: () => number | null;
};

const NodeContext = createContext<NodeContextState | null>(null);

// ─────────────────────────────────────────────────────────────────────────────
// Port Definitions
// ─────────────────────────────────────────────────────────────────────────────

const SOURCE_PORTS: PortDefinition[] = [
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
  if (node.type === "source") {
    const d = node.data as SourceFlowNodeData;
    return d.variant === "number" ? d.value : undefined;
  }
  const d = node.data as OperatorFlowNodeData;
  return d.result;
}

function getRightmostEvaluableNode(nodes: DataflowNode[]): DataflowNode | null {
  const evalNodes = nodes.filter(
    (n): n is Extract<DataflowNode, { type: "source" | "operator" }> =>
      n.type === "source" || n.type === "operator"
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


// ─────────────────────────────────────────────────────────────────────────────
// Provider
// ─────────────────────────────────────────────────────────────────────────────

type NodeProviderProps = {
  children: ReactNode;
  flowContainerRef: React.RefObject<HTMLDivElement | null>;
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

  // Program executor with lifecycle management
  const executorRef = useRef<ProgramExecutor | null>(null);

  // Create executor on mount, reset on unmount
  useEffect(() => {
    executorRef.current = createProgramExecutor();
    console.log("[NodeProvider] Interpreter created");

    return () => {
      executorRef.current?.reset();
      executorRef.current = null;
      console.log("[NodeProvider] Interpreter destroyed");
    };
  }, []);

  const [nodes, setNodes, onNodesChange] = useNodesState<DataflowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedPort, setSelectedPort] = useState<PortIdentifier | null>(null);

  // Estado de ejecución
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<number | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);

  // Sync vision cards to nodes
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
            type: "source" as const,
            position,
            data: {
              variant: "number",
              value: parsed.value,
              trackId: c.trackId,
            },
          });
          idx++;
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
          idx++;
          continue;
        }

        if (parsed.type === "operatorCanvas") {
          additions.push({
            id: nodeId,
            type: "operator" as const,
            position,
            data: {
              operator: parsed.operator,
              trackId: c.trackId,
            },
          });
          idx++;
          continue;
        }

        if (parsed.type === "programResultCard") {
          additions.push({
            id: nodeId,
            type: "programOutput" as const,
            position,
            data: {},
          });
          idx++;
          continue;
        }

        if (parsed.type === "deckShape") {
          additions.push({
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "shape",
              yoloClass: parsed.yoloClass,
              shape: parsed.shape,
              size: parsed.size,
              color: parsed.color,
              trackId: c.trackId,
            },
          });
          idx++;
          continue;
        }

        if (parsed.type === "deckFood") {
          additions.push({
            id: nodeId,
            type: "source" as const,
            position,
            data: {
              variant: "food",
              yoloClass: parsed.yoloClass,
              food: parsed.food,
              trackId: c.trackId,
            },
          });
          idx++;
          continue;
        }

        // Tipo desconocido: mostrar como número con subtítulo
        additions.push({
          id: nodeId,
          type: "source" as const,
          position,
          data: {
            variant: "number",
            value: 0,
            visionSubtitle: parsed.label,
            trackId: c.trackId,
          },
        });
        idx++;
      }

      if (grapesFlowPos) {
        additions.push({
          id: VISION_PROGRAM_OUTPUT_ID,
          type: "programOutput" as const,
          position: {
            x: grapesFlowPos.x + VISION_CARD_BOX + VISION_RESULT_GAP,
            y: grapesFlowPos.y,
          },
          data: { value: preservedValue },
        });
      }

      return [...withoutLive, ...additions];
    });
  }, [visionSyncEnabled, lastCardFrame, setNodes, flowContainerRef]);

  // Clean up orphan edges when nodes change
  useEffect(() => {
    const nodeIds = new Set(nodes.map((n) => n.id));

    setEdges((eds) => {
      const validEdges = eds.filter(
        (e) => nodeIds.has(e.source) && nodeIds.has(e.target)
      );
      // Only update if something was removed
      return validEdges.length === eds.length ? eds : validEdges;
    });
  }, [nodes, setEdges]);


  // Automatic interpreter execution when nodes/edges change
  useEffect(() => {
    const evalNodes = nodes.filter(
      (n) => n.type === "source" || n.type === "operator"
    );

    // No ejecutar si no hay nodos evaluables o no hay executor
    if (evalNodes.length === 0 || !executorRef.current) return;

    // Ejecutar el programa con el interpreter
    executorRef.current.execute(nodes, edges).then((result) => {
      if (result.success && result.results) {
        setExecutionError(null);

        // Actualizar cada programOutput con su resultado específico
        setNodes((nds) => {
          let changed = false;
          const updated = nds.map((n) => {
            if (n.type !== "programOutput") return n;
            const resultValue = result.results!.get(n.id);
            if (resultValue === undefined) return n;

            // Extraer value y description según el tipo de resultado
            const currentData = n.data as ProgramOutputFlowNodeData;
            let newData: ProgramOutputFlowNodeData;

            if (resultValue.kind === "number") {
              newData = { value: resultValue.value, description: undefined };
            } else {
              // kind === "semantic"
              newData = {
                value: resultValue.result.totalAmount,
                description: resultValue.result.description,
              };
            }

            // Solo actualizar si cambió
            if (
              currentData.value === newData.value &&
              currentData.description === newData.description
            ) {
              return n;
            }

            changed = true;
            return { ...n, data: { ...n.data, ...newData } };
          });
          return changed ? updated : nds;
        });
      } else if (result.error) {
        setExecutionResult(null);
        setExecutionError(result.error);
      }
    });
  }, [nodes, edges, setNodes]);

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

  const getNodePorts = useCallback((nodeType: "source" | "operator") => {
    return nodeType === "source" ? SOURCE_PORTS : OPERATOR_PORTS;
  }, []);

  const addNumberNode = useCallback(
    (value: number, position?: { x: number; y: number }) => {
      const id = `num${value}_${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: "source" as const,
          position: position ?? {
            x: 100 + (nds.length % 3) * 60,
            y: 80 + Math.floor(nds.length / 3) * 100,
          },
          data: { variant: "number", value },
        },
      ]);
    },
    [setNodes]
  );

  const addOperatorNode = useCallback(
    (operator: OperatorType, position?: { x: number; y: number }) => {
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

  const addResultAnchorPair = useCallback(() => {
    setNodes((nds) => {
      const pairId = `manual_out_${Date.now()}`;
      return [
        ...nds,
        { id: `${pairId}`, type: "programOutput" as const, position: { x: 304, y: 120 }, data: {} },
      ];
    });
  }, [setNodes]);

  const addResultCard = useCallback(() => {
    setNodes((nds) => [
      ...nds,
      {
        id: `result_${Date.now()}`,
        type: "programOutput" as const,
        position: { x: 380 + (nds.length % 4) * 220, y: 140 },
        data: {},
      },
    ]);
  }, [setNodes]);

  const spawnDeckYoloClass = useCallback(
    (yoloClass: string) => {
      const spawn = spawnActionForYoloClass(yoloClass);
      if (!spawn) return;
      if (spawn.kind === "number") return addNumberNode(spawn.value);
      if (spawn.kind === "operator") return addOperatorNode(spawn.operator);
      if (spawn.kind === "resultCard") return addResultCard();
      if (spawn.kind === "shape") {
        setNodes((nds) => [
          ...nds,
          {
            id: `deck_${spawn.yoloClass}_${Date.now()}`,
            type: "source" as const,
            position: { x: 120, y: 200 + (nds.length % 6) * 28 },
            data: {
              variant: "shape",
              yoloClass: spawn.yoloClass,
              shape: spawn.shape,
              size: spawn.size,
              color: spawn.color,
            },
          },
        ]);
        return;
      }
      if (spawn.kind === "food") {
        setNodes((nds) => [
          ...nds,
          {
            id: `deck_${spawn.yoloClass}_${Date.now()}`,
            type: "source" as const,
            position: { x: 120, y: 200 + (nds.length % 6) * 28 },
            data: { variant: "food", yoloClass: spawn.yoloClass, food: spawn.food },
          },
        ]);
      }
    },
    [addNumberNode, addOperatorNode, addResultCard, setNodes]
  );

  const executeProgram = useCallback(async () => {
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

      if (result.success && result.result !== undefined) {
        setExecutionResult(result.result);
        setExecutionError(null);
        syncProgramOutputValue(result.result);

        // Log stats for debugging
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
