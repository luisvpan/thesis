import { useEffect, useMemo, useCallback } from 'react';
import type { Node, NodeProps, Edge } from '@xyflow/react';
import { Position, useReactFlow } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import {
  type OperatorType,
  type DivisionMode,
  isFilterOperatorType,
  isOrderOperatorType,
} from '@/types/card-types';
import type { OrderCriterio } from '@/data/yoloDeckCatalog';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { useNode } from '@/contexts/NodeContext';
import type { DataflowNode, PortKindInfo } from '@/contexts/node/types';
import type { HandleKind } from './handle-kinds';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';
import type { SourceFlowNodeData } from './SourceFlowNode';
import type { ProgramOutputFlowNodeData } from './ProgramOutputFlowNode';

export type OperatorFlowNodeData = VisionNodeMeta &
  Pick<
    ProgramOutputFlowNodeData,
    | 'value'
    | 'description'
    | 'visualStrip'
    | 'originalElements'
    | 'isSingleCpaObject'
    | 'singleCpaObjectMeta'
    | 'numerator'
    | 'denominator'
  > & {
    operator: OperatorType;
    /** Resumen numérico en la carta del operador (sincronizado con `value`). */
    result?: number;
    /** Modo de visualización para división: partitivo o cuotativo. Solo aplica cuando operator === 'division'. */
    divisionMode?: DivisionMode;
    /** Criterio implícito para operadores de ordenamiento (ej: smallest_to_largest tiene criterio size). */
    criterio?: OrderCriterio;
  };

export type OperatorFlowNode = Node<OperatorFlowNodeData, 'operator'>;

function operatorSymbol(operator: OperatorType): string {
  if (operator === 'adicion') return '+';
  if (operator === 'sustraccion') return '-';
  if (operator === 'multiplicacion') return '*';
  if (operator === 'division') return '/';
  if (operator === 'comparar') return '=?';
  return operator;
}

/**
 * Determine what kinds a handle accepts based on the operator type.
 * - Filter operators: input "a" (items) accepts cpa
 * - Filter operators: input "b" (criterion) accepts only keyword
 * - All other operators: accept cpa and rational
 */
function getHandleAccepts(operator: OperatorType, handleId: string): HandleKind[] {
  if (isOrderOperatorType(operator)) {
    return ['group'];
  }
  if (isFilterOperatorType(operator)) {
    if (handleId === 'b') {
      return ['keyword'];
    }
    return ['cpa'];
  }
  return ['cpa', 'rational'];
}

/**
 * Hook to compute the output kind dynamically based on connected inputs.
 * Returns 'cpa' if any input is cpa, otherwise 'rational'.
 */
function useOutputKind(
  nodeId: string,
  operator: OperatorType,
  edges: Edge[],
  getPortKindInfo: (nodeId: string, handleId: string) => PortKindInfo | undefined
): HandleKind {
  return useMemo(() => {
    if (isOrderOperatorType(operator)) {
      return 'group';
    }
    const inputEdges = edges.filter((e) => e.target === nodeId);
    for (const edge of inputEdges) {
      const sourceInfo = getPortKindInfo(edge.source, edge.sourceHandle ?? 'out');
      if (sourceInfo?.produces === 'cpa') {
        return 'cpa';
      }
    }
    return 'rational';
  }, [nodeId, operator, edges, getPortKindInfo]);
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
  const isOrderOp = isOrderOperatorType(operator);
  const { registerPortKind, unregisterPortKinds, nodes, edges, getPortKindInfo } = useNode();
  const { setNodes } = useReactFlow();
  const shellClass = useFlowNodeShellClass();

  const acceptsA = getHandleAccepts(operator, 'a');
  const acceptsB = getHandleAccepts(operator, 'b');
  const producesOut = useOutputKind(id, operator, edges, getPortKindInfo);

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
    registerPortKind(id, 'a', { accepts: acceptsA });
    if (!isOrderOp) {
      registerPortKind(id, 'b', { accepts: acceptsB });
    }
    registerPortKind(id, 'out', { produces: producesOut });
    return () => unregisterPortKinds(id);
  }, [id, operator, isOrderOp, acceptsA, acceptsB, producesOut, registerPortKind, unregisterPortKinds]);

  return (
    <div className={`relative h-52 w-30 -translate-x-[15%] -translate-y-[45%] ${shellClass}`}>
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
        handleVariant="operator-in-a"
        accepts={acceptsA}
        style={{
          top: isOrderOp ? '50%' : '25%',
          transform: 'translateX(-100px)',
        }}
      />
      {!isOrderOp ? (
        <ClickableHandle
          type="target"
          position={Position.Left}
          id="b"
          nodeId={id}
          handleVariant="operator-in-b"
          accepts={acceptsB}
          style={{ top: '75%', transform: 'translateX(-100px)' }}
        />
      ) : null}
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
        handleVariant="operator-out"
        produces={producesOut}
        style={{ transform: 'translateX(100px)' }}
      />
    </div>
  );
}
