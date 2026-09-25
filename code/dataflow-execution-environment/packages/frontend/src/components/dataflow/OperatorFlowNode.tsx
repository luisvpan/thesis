import { useEffect, useMemo } from 'react';
import type { Node, NodeProps, Edge } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import {
  type OperatorType,
  isFilterOperatorType,
  isMathOperatorType,
  isSingleInputOperatorType,
} from '@/types/card-types';
import type { OrderCriterio } from '@/data/yoloDeckCatalog';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import type { NodeErrorMark } from '@/contexts/node/errorMarks';
import { useNode } from '@/contexts/NodeContext';
import type { PortKindInfo } from '@/contexts/node/types';
import type { HandleKind } from './handle-kinds';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';
import type { ResultValue } from '@/services/executeProgram';
import { numericValueOf, type WithResultValue } from '@/utils/resultValueDisplay';

export type OperatorFlowNodeData = VisionNodeMeta &
  WithResultValue & {
    operator: OperatorType;
    /** Criterio implícito para operadores de ordenamiento (ej: smallest_to_largest tiene criterio size). */
    criterio?: OrderCriterio;
    /** Papel de la carta en el error de una salida, si lo tiene (§4). */
    errorMark?: NodeErrorMark;
  };

export type OperatorFlowNode = Node<OperatorFlowNodeData, 'operator'>;

/** Lo que la carta del operador dice de su resultado, en una línea. */
function operatorSubtitle(result: ResultValue | undefined): string {
  if (!result) return 'esperando entradas';
  if (result.kind === 'boolean') return result.value ? 'verdadero' : 'falso';

  const value = numericValueOf(result);
  return value !== undefined ? `resultado: ${value}` : 'listo';
}

const ORDER_PROPERTY_LABEL: Record<string, string> = {
  quantity: 'cantidad',
  size: 'tamaño',
  color: 'color',
  subtype: 'forma',
};

/**
 * Hay cuatro cartas de orden repartidas en dos operadores: lo que las distingue
 * es por qué propiedad ordenan, así que el rótulo lo dice.
 */
function operatorTitle(operator: OperatorType, criterio?: { property: string }): string {
  if (!criterio) return operator;
  return `${operator} · ${ORDER_PROPERTY_LABEL[criterio.property] ?? criterio.property}`;
}

function operatorSymbol(operator: OperatorType): string {
  if (operator === 'adicion') return '+';
  if (operator === 'sustraccion') return '-';
  if (operator === 'multiplicacion') return '*';
  if (operator === 'division') return '/';
  if (operator === 'comparar') return '=?';
  if (operator === 'primero') return '1°';
  if (operator === 'ultimo') return 'Últ';
  if (operator === 'contar') return '#';
  return operator;
}

/**
 * Determine what kinds a handle accepts based on the operator type.
 * - Math operators (+ − × ÷): groups, CPA objects, and numbers; not criteria keywords
 * - Filter operators: input "a" (items) accepts groups and CPA
 * - Filter operators: input "b" (criterion) accepts criteria keywords
 * - Other operators: CPA and rational only
 */
function getHandleAccepts(operator: OperatorType, handleId: string): HandleKind[] {
  if (isSingleInputOperatorType(operator)) {
    return ['group', 'cpa'];
  }
  if (isFilterOperatorType(operator)) {
    if (handleId === 'b') {
      return ['keyword'];
    }
    return ['group', 'cpa'];
  }
  if (isMathOperatorType(operator)) {
    return ['group', 'cpa', 'rational'];
  }
  if (operator === 'comparar') {
    return ['group', 'cpa', 'rational'];
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
    if (isSingleInputOperatorType(operator)) {
      if (operator === 'contar') return 'rational';
      const inputEdges = edges.filter((e) => e.target === nodeId);
      for (const edge of inputEdges) {
        const sourceInfo = getPortKindInfo(edge.source, edge.sourceHandle ?? 'out');
        if (sourceInfo?.produces === 'group') return 'group';
        if (sourceInfo?.produces === 'cpa') return 'cpa';
      }
      return 'group';
    }
    if (operator === 'comparar') return 'rational';
    const inputEdges = edges.filter((e) => e.target === nodeId);
    for (const edge of inputEdges) {
      if (isFilterOperatorType(operator) && edge.targetHandle === 'b') {
        continue;
      }
      const sourceInfo = getPortKindInfo(edge.source, edge.sourceHandle ?? 'out');
      if (sourceInfo?.produces === 'group') {
        return 'group';
      }
      if (sourceInfo?.produces === 'cpa') {
        return 'cpa';
      }
    }
    return 'rational';
  }, [nodeId, operator, edges, getPortKindInfo]);
}

export function OperatorFlowNode({ id, data }: NodeProps<OperatorFlowNode>) {
  const d = (data ?? {}) as OperatorFlowNodeData;
  const operator = d.operator ?? 'adicion';
  const isSingleInputOp = isSingleInputOperatorType(operator);
  const { registerPortKind, unregisterPortKinds, edges, getPortKindInfo } = useNode();
  const shellClass = useFlowNodeShellClass();

  const acceptsA = getHandleAccepts(operator, 'a');
  const acceptsB = getHandleAccepts(operator, 'b');
  const producesOut = useOutputKind(id, operator, edges, getPortKindInfo);

  // Register port kinds when component mounts or operator/output changes
  useEffect(() => {
    registerPortKind(id, 'a', { accepts: acceptsA });
    if (!isSingleInputOp) {
      registerPortKind(id, 'b', { accepts: acceptsB });
    }
    registerPortKind(id, 'out', { produces: producesOut });
    return () => unregisterPortKinds(id);
  }, [id, operator, isSingleInputOp, acceptsA, acceptsB, producesOut, registerPortKind, unregisterPortKinds]);

  return (
    <div className={`relative h-52 w-30 -translate-x-[15%] -translate-y-[45%] ${shellClass}`}>
      <TrackIdBadge trackId={readTrackId(d)} />
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="a"
        nodeId={id}
        handleVariant="operator-in-a"
        accepts={acceptsA}
        style={{
          top: isSingleInputOp ? '50%' : '25%',
          transform: 'translateX(-100px)',
        }}
      />
      {!isSingleInputOp ? (
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
        errorMark={d.errorMark}
        title={operatorTitle(operator, d.criterio)}
        content={<span className="text-xs font-black text-slate-100">{operatorSymbol(operator)}</span>}
        subtitle={operatorSubtitle(d.resultValue)}
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
