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
import type { HandleKind, HandleAcceptance } from './handle-kinds';
import type { PortKindInfo } from '@/contexts/node/types';
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
 * - Filter operators: input "a" (items) prefers cpa, tolerates rational
 * - Filter operators: input "b" (criterion) accepts only keyword
 * - All other operators: accept cpa and rational as primary
 */
function getHandleAcceptance(operator: OperatorType, handleId: string): HandleAcceptance {
  if (isFilterOperatorType(operator)) {
    if (handleId === 'b') {
      return { primary: ['keyword'] };
    }
    // items: cpa preferred, rational tolerated
    return { primary: ['cpa'], tolerated: ['rational'] };
  }
  // Arithmetic and Order operators: both kinds are primary
  return { primary: ['cpa', 'rational'] };
}

/**
 * Hook to compute the output kind dynamically based on connected inputs.
 * Returns 'cpa' if any input is cpa, otherwise 'rational'.
 */
function useOutputKind(
  nodeId: string,
  edges: Edge[],
  getPortKindInfo: (nodeId: string, handleId: string) => PortKindInfo | undefined
): HandleKind {
  return useMemo(() => {
    const inputEdges = edges.filter((e) => e.target === nodeId);
    for (const edge of inputEdges) {
      const sourceInfo = getPortKindInfo(edge.source, edge.sourceHandle ?? 'out');
      if (sourceInfo?.produces === 'cpa') {
        return 'cpa';
      }
    }
    return 'rational';
  }, [nodeId, edges, getPortKindInfo]);
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
  const { registerPortKind, unregisterPortKinds, nodes, edges, getPortKindInfo } = useNode();
  const { setNodes } = useReactFlow();

  // Determine acceptance for each input handle
  const acceptanceA = getHandleAcceptance(operator, 'a');
  const acceptanceB = getHandleAcceptance(operator, 'b');
  // Output kind computed dynamically based on connected inputs
  const producesOut = useOutputKind(id, edges, getPortKindInfo);

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

  // Register port kinds when component mounts or operator/output changes
  useEffect(() => {
    registerPortKind(id, 'a', { acceptance: acceptanceA });
    registerPortKind(id, 'b', { acceptance: acceptanceB });
    registerPortKind(id, 'out', { produces: producesOut });
    return () => unregisterPortKinds(id);
  }, [id, operator, acceptanceA, acceptanceB, producesOut, registerPortKind, unregisterPortKinds]);

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
        acceptance={acceptanceA}
        style={{ top: '25%', transform: 'translateX(-100px)' }}
      />
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="b"
        nodeId={id}
        acceptance={acceptanceB}
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

