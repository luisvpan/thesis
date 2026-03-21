import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  addEdge,
  type Connection,
  type Edge,
  type Node,
  Background,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { NumberFlowNode, OperatorFlowNode } from '@/components/dataflow';
import type { NumberFlowNodeData, OperatorFlowNodeData } from '@/components/dataflow';
import type { MathOperatorType } from '@/types/card-types';
import type { SocketAddNodePayload, SocketAddEdgePayload } from '@/types/socket-types';
import { useSocket } from '@/contexts/SocketContext';
import { useVision } from '@/contexts/VisionContext';
import { VisionDetectedBadge } from '@/components/VisionDetectedBadge';
import { getLevelConfig } from '@/data/levelConfig';
import { ArrowLeft, Plus, Minus, Volume2, Play } from 'lucide-react';

type ViewMode = 'pictorico' | 'concreto' | 'abstracto';

const VIEW_MODE_LABELS: Record<ViewMode, string> = {
  pictorico: 'Pictórico',
  concreto: 'Concreto',
  abstracto: 'Abstracto',
};

type DataflowNode = Node<NumberFlowNodeData, 'number'> | Node<OperatorFlowNodeData, 'operator'>;

const nodeTypes: NodeTypes = {
  number: NumberFlowNode,
  operator: OperatorFlowNode,
};

/** Valor de salida de un nodo: número tiene value, operador tiene value (resultado) */
function getNodeValue(node: DataflowNode | null | undefined): number | undefined {
  if (!node?.data) return undefined;
  const d = node.data as NumberFlowNodeData & OperatorFlowNodeData;
  return d.value ?? d.result;
}

/** Nodo con mayor position.x (el más a la derecha del canvas) */
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
  const edgeA = edgesToOperator.find((e) => e.targetHandle === 'a');
  const edgeB = edgesToOperator.find((e) => e.targetHandle === 'b');
  const nodeA = edgeA ? nodes.find((n) => n.id === edgeA.source) : null;
  const nodeB = edgeB ? nodes.find((n) => n.id === edgeB.source) : null;
  const valA = getNodeValue(nodeA);
  const valB = getNodeValue(nodeB);
  if (typeof valA !== 'number' || typeof valB !== 'number') return undefined;
  const operator = (nodes.find((n) => n.id === operatorId)?.data as OperatorFlowNodeData | undefined)?.operator;
  if (operator === 'adicion') return valA + valB;
  if (operator === 'sustraccion') return valA - valB;
  return undefined;
}

const initialNodes: DataflowNode[] = [
  {
    id: 'num-3',
    type: 'number',
    position: { x: 80, y: 80 },
    data: { value: 3 },
  },
  {
    id: 'num-5',
    type: 'number',
    position: { x: 80, y: 200 },
    data: { value: 5 },
  },
  {
    id: 'op-sum',
    type: 'operator',
    position: { x: 320, y: 120 },
    data: { operator: 'adicion' },
  },
];

const initialEdges: Edge[] = [];

function speakTitle(title: string, subtitle: string) {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(`${title}. ${subtitle}`);
  u.lang = 'es-ES';
  window.speechSynthesis.speak(u);
}

export default function DataflowPage({ isSandbox }: { isSandbox: boolean }) {
  const params = useParams();
  const worldId = params.worldId;
  const level = params.level;
  const levelConfig = getLevelConfig(worldId, level, isSandbox);
  const socket = useSocket();
  const { last: visionLast } = useVision();
  const lastVisionIngestRef = useRef<number | null>(null);
  const visionLayoutIndexRef = useRef(0);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [viewMode, setViewMode] = useState<ViewMode>('pictorico');
  const [executedResult, setExecutedResult] = useState<number | null>(null);
  const backTo = worldId ? (isSandbox ? '/juego' : `/juego/${worldId}`) : '/';

  useEffect(() => {
    if (!socket) return;

    const onAddNode = (payload: SocketAddNodePayload) => {
      const node: DataflowNode = {
        id: payload.id,
        type: payload.type,
        position: payload.position,
        data: payload.data as NumberFlowNodeData & OperatorFlowNodeData,
      };
      setNodes((nds) => [...nds, node]);
    };

    const onAddEdge = (payload: SocketAddEdgePayload) => {
      const edge: Edge = {
        id: payload.id ?? `e-${payload.source}-${payload.target}-${Date.now()}`,
        source: payload.source,
        target: payload.target,
        sourceHandle: payload.sourceHandle ?? undefined,
        targetHandle: payload.targetHandle ?? undefined,
      };
      setEdges((eds) => [...eds, edge]);
    };

    socket.on('addNode', onAddNode);
    socket.on('addEdge', onAddEdge);
    return () => {
      socket.off('addNode', onAddNode);
      socket.off('addEdge', onAddEdge);
    };
  }, [socket, setNodes, setEdges]);

  /**
   * Cada detección de visión con valor semántico (dígito 1..9) añade un nodo número.
   * Clases que no son one..nine solo actualizan el panel; no crean nodo.
   */
  useEffect(() => {
    if (!visionLast) return;
    if (lastVisionIngestRef.current === visionLast.t) return;
    lastVisionIngestRef.current = visionLast.t;

    if (visionLast.number === undefined || visionLast.number === null) return;

    const idx = visionLayoutIndexRef.current++;
    const value = visionLast.number;
    const id = `vision-${visionLast.t}`;

    setNodes((nds) => [
      ...nds,
      {
        id,
        type: 'number' as const,
        position: {
          x: 40 + (idx % 5) * 92,
          y: 300 + Math.floor(idx / 5) * 104,
        },
        data: { value },
      },
    ]);
  }, [visionLast, setNodes]);

  const cycleViewMode = useCallback(() => {
    setViewMode((m) => (m === 'pictorico' ? 'concreto' : m === 'concreto' ? 'abstracto' : 'pictorico'));
  }, []);

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge(connection, eds)),
    [setEdges]
  );

  // Calcular resultados de operadores (varios pasos para cadenas op1 → op2)
  useEffect(() => {
    setNodes((nds) => {
      let current = nds;
      for (let pass = 0; pass < 10; pass++) {
        let changed = false;
        current = current.map((node) => {
          if (node.type !== 'operator') return node;
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

  const addNumberNode = useCallback(
    (value: number) => {
      const id = `num-${value}-${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: 'number',
          position: { x: 100 + (nds.length % 3) * 60, y: 80 + Math.floor(nds.length / 3) * 100 },
          data: { value },
        },
      ]);
    },
    [setNodes]
  );

  const addOperatorNode = useCallback(
    (operator: MathOperatorType) => {
      const id = `op-${operator}-${Date.now()}`;
      setNodes((nds) => [
        ...nds,
        {
          id,
          type: 'operator',
          position: { x: 320 + (nds.filter((n) => n.type === 'operator').length % 2) * 200, y: 120 },
          data: { operator },
        },
      ]);
    },
    [setNodes]
  );

  const onExecute = useCallback(() => {
    const rightmost = getRightmostNode(nodes);
    const value = rightmost ? getNodeValue(rightmost) : undefined;
    setExecutedResult(typeof value === 'number' ? value : null);
  }, [nodes]);

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-900">
      {/* Header: modo oscuro */}
      <header className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700 shrink-0 gap-4">
        <Link
          to={backTo}
          className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors shrink-0 text-lg"
        >
          <ArrowLeft className="w-5 h-5" />
          Volver
        </Link>

        <div className="flex-1 flex flex-col items-center justify-center min-w-0">
          <h1 className="text-2xl font-bold text-white text-center truncate max-w-full">
            {levelConfig.title}
          </h1>
          <p className="text-lg text-slate-400 text-center truncate max-w-full">
            {levelConfig.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
          <VisionDetectedBadge />
          <button
            type="button"
            onClick={onExecute}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-teal-500 hover:bg-teal-600 text-white text-lg font-medium transition-colors"
            title="Ejecutar"
          >
            <Play className="w-4 h-4" />
            Ejecutar
          </button>
          <button
            type="button"
            onClick={cycleViewMode}
            className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-lg font-medium transition-colors border border-slate-600"
          >
            {VIEW_MODE_LABELS[viewMode]}
          </button>
          <button
            type="button"
            onClick={() => speakTitle(levelConfig.title, levelConfig.subtitle)}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-white transition-colors border border-slate-600"
            title="Reproducir título"
          >
            <Volume2 className="w-5 h-5" />
          </button>
        </div>
      </header>

      <div className="flex-1 flex min-h-0">
        {!isSandbox && (
          <>
            {/* Sidebar: modo oscuro */}
            <aside className="w-64 shrink-0 flex flex-col bg-slate-800 border-r border-slate-700 overflow-y-auto">
              <section className="p-3 border-b border-slate-700">
                <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
                  Mochila
                </h2>
                <div className="space-y-3">
                  <div>
                    <p className="text-base font-medium text-slate-400 mb-2">Añadir número</p>
                    <div className="flex flex-wrap gap-1">
                      {levelConfig.numbers.map((n) => (
                        <button
                          key={n}
                          type="button"
                          onClick={() => addNumberNode(n)}
                          className="w-9 h-9 rounded-lg bg-slate-700 hover:bg-teal-500 text-slate-200 hover:text-white font-bold text-lg transition-colors"
                        >
                          {n}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-base font-medium text-slate-400 mb-2">Añadir operador</p>
                    <div className="flex gap-2">
                      {levelConfig.operators.includes('adicion') && (
                        <button
                          type="button"
                          onClick={() => addOperatorNode('adicion')}
                          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-teal-500 hover:bg-teal-600 text-white font-bold text-lg transition-colors"
                        >
                          <Plus className="w-4 h-4" />
                          Suma
                        </button>
                      )}
                      {levelConfig.operators.includes('sustraccion') && (
                        <button
                          type="button"
                          onClick={() => addOperatorNode('sustraccion')}
                          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-bold text-lg transition-colors"
                        >
                          <Minus className="w-4 h-4" />
                          Resta
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </section>
              <section className="p-3 flex-1">
                <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
                  Reglas para el lenguaje
                </h2>
                <p className="text-slate-400 text-base leading-relaxed">
                  {levelConfig.rule}
                </p>
              </section>
            </aside>
          </>
        )}

        {/* Área principal: canvas ReactFlow (fondo negro) */}
        <div className="flex-1 relative min-w-0 bg-black">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            className="bg-black"
            minZoom={1}
            maxZoom={1}
            zoomOnScroll={false}
            zoomOnPinch={false}
            zoomOnDoubleClick={false}
            panOnDrag={false}
            panOnScroll={false}
            autoPanOnNodeDrag={false}
          >
            <Background color="#334155" gap={16} size={0.5} />
          </ReactFlow>
        </div>
      </div>

      {/* Footer: modo oscuro */}
      <footer className="shrink-0 bg-slate-800 border-t border-slate-700 px-4 py-3">
        <section>
          <p className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-1">
            El resultado se mostrará acá
          </p>
          <p className="text-2xl font-semibold text-white min-h-[1.5rem]">
            {executedResult !== null ? executedResult : '—'}
          </p>
        </section>
      </footer>
    </div>
  );
}
