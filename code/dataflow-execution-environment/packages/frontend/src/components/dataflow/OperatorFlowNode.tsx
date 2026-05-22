import { useEffect } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import { type OperatorType, isFilterOperatorType } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { useNode } from '@/contexts/NodeContext';
import type { HandleKind } from './handle-kinds';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';

export type OperatorFlowNodeData = VisionNodeMeta & {
  operator: OperatorType;
  result?: number;
  value?: number;
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

export function OperatorFlowNode({ id, data }: NodeProps<OperatorFlowNode>) {
  const d = (data ?? {}) as OperatorFlowNodeData;
  const operator = d.operator ?? 'adicion';
  const { registerPortKind, unregisterPortKinds } = useNode();
  const shellClass = useFlowNodeShellClass();

  // Determine accepts for each input handle
  const acceptsA = getHandleAccepts(operator, 'a');
  const acceptsB = getHandleAccepts(operator, 'b');
  // Output always produces "any" (could be CPA or rational depending on inputs)
  const producesOut: HandleKind = 'any';

  // Register port kinds when component mounts or operator changes
  useEffect(() => {
    registerPortKind(id, 'a', { accepts: acceptsA });
    registerPortKind(id, 'b', { accepts: acceptsB });
    registerPortKind(id, 'out', { produces: producesOut });
    return () => unregisterPortKinds(id);
  }, [id, operator, acceptsA, acceptsB, producesOut, registerPortKind, unregisterPortKinds]);

  return (
    <div className={`relative h-52 w-30 -translate-x-[15 %] -translate-y-[45%] ${shellClass}`}>
      <TrackIdBadge trackId={readTrackId(d)} />
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

