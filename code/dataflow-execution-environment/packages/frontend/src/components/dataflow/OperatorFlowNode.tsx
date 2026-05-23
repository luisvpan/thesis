import { useEffect, useMemo, useCallback } from 'react';
import type { Node, NodeProps, Edge } from '@xyflow/react';
import { Position, useReactFlow } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import { type OperatorType, type DivisionMode, isFilterOperatorType } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { useNode } from '@/contexts/NodeContext';
import type { DataflowNode } from '@/contexts/node/types';
import type { HandleKind } from './handle-kinds';
import type { SourceFlowNodeData } from './SourceFlowNode';

export type OperatorFlowNodeData = VisionNodeMeta & {
  operator: OperatorType;
  result?: number;
  value?: number;
  /** Modo de visualización para división: partitivo o cuotativo. Solo aplica cuando operator === 'division'. */
  divisionMode?: DivisionMode;
};

export type OperatorFlowNode = Node<OperatorFlowNodeData, 'operator'>;

function operatorSymbol(operator: OperatorType): string {
  if (operator === 'adicion') return '+';
  if (operator === 'sustraccion') return '-';
  if (operator === 'multiplicacion') return '*';
  if (operator === 'division') return '/';
  return operator;
}

/**
 * Determine what kinds a handle accepts based on the operator type.
 * - Filter operators: input "b" (criterion) accepts only "keyword"
 * - All other operators: accept "any"
 */
function getHandleAccepts(operator: OperatorType, handleId: string): HandleKind[] {
  if (isFilterOperatorType(operator)) {
    // For filter: "a" accepts items (any), "b" accepts criterion (keyword)
    return handleId === 'b' ? ['keyword'] : ['any'];
  }
  // Math and Order operators: both inputs accept any type
  return ['any'];
}

/**
 * Determina si el input "a" de una división está conectado a un CPA (no a un número).
 * Usado para mostrar/ocultar el toggle partitivo/cuotativo.
 */
function useDivisionHasCpaInput(
  nodeId: string,
  operator: OperatorType,
  nodes: DataflowNode[],
  edges: Edge[]
): boolean {
  return useMemo(() => {
    if (operator !== 'division') return false;
    const aEdge = edges.find((e) => e.target === nodeId && e.targetHandle === 'a');
    if (!aEdge) return false;
    const aSource = nodes.find((n) => n.id === aEdge.source);
    if (!aSource) return false;
    if (aSource.type === 'source') {
      const d = aSource.data as SourceFlowNodeData;
      return d.variant !== 'number';
    }
    return aSource.type === 'operator' || aSource.type === 'arrayClose';
  }, [nodeId, operator, nodes, edges]);
}

export function OperatorFlowNode({ id, data }: NodeProps<OperatorFlowNode>) {
  const d = (data ?? {}) as OperatorFlowNodeData;
  const operator = d.operator ?? 'adicion';
  const { registerPortKind, unregisterPortKinds, nodes, edges } = useNode();
  const { setNodes } = useReactFlow();

  // Determine accepts for each input handle
  const acceptsA = getHandleAccepts(operator, 'a');
  const acceptsB = getHandleAccepts(operator, 'b');
  // Output always produces "any" (could be CPA or rational depending on inputs)
  const producesOut: HandleKind = 'any';

  // Division toggle: only show when input "a" is CPA (not a number)
  const showDivisionToggle = useDivisionHasCpaInput(id, operator, nodes, edges);
  const divisionMode = d.divisionMode ?? 'partitivo';

  const toggleDivisionMode = useCallback(() => {
    setNodes((nds) =>
      nds.map((n) =>
        n.id === id
          ? { ...n, data: { ...n.data, divisionMode: divisionMode === 'partitivo' ? 'cuotativo' : 'partitivo' } }
          : n
      )
    );
  }, [id, divisionMode, setNodes]);

  // Register port kinds when component mounts or operator changes
  useEffect(() => {
    registerPortKind(id, 'a', { accepts: acceptsA });
    registerPortKind(id, 'b', { accepts: acceptsB });
    registerPortKind(id, 'out', { produces: producesOut });
    return () => unregisterPortKinds(id);
  }, [id, operator, acceptsA, acceptsB, producesOut, registerPortKind, unregisterPortKinds]);

  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[25%]">
      <TrackIdBadge trackId={readTrackId(d)} />
      {/* Toggle partitivo/cuotativo para divisiones CPA */}
      {showDivisionToggle && (
        <button
          type="button"
          onClick={toggleDivisionMode}
          className="absolute -top-6 left-1/2 -translate-x-1/2 px-2 py-0.5 text-[10px] font-semibold tracking-wide rounded bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors z-10"
          title={divisionMode === 'partitivo' ? 'Dividir en N grupos' : 'Grupos de tamaño N'}
        >
          {divisionMode === 'partitivo' ? 'Partitiva' : 'Cuotativa'}
        </button>
      )}
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="a"
        nodeId={id}
        accepts={acceptsA}
        style={{ top: '25%', transform: 'translateX(-100px)' }}
      />
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="b"
        nodeId={id}
        accepts={acceptsB}
        style={{ top: '75%', transform: 'translateX(-100px)' }}
      />
      <FlowNodeCard
        family="transformation"
        title={operator}
        content={<span className="text-xs font-black text-slate-100">{operatorSymbol(operator)}</span>}
        subtitle={d.result !== undefined ? `resultado: ${d.result}` : 'esperando entradas'}
      />
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
        produces={producesOut}
        style={{ transform: 'translateX(100px)' }}
      />
    </div>
  );
}

