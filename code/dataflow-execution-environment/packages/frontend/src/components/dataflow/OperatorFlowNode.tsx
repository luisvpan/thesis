import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { OperatorType } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';

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

export function OperatorFlowNode({ id, data }: NodeProps<OperatorFlowNode>) {
  const d = (data ?? {}) as OperatorFlowNodeData;
  const operator = d.operator ?? 'adicion';

  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[25%]">
      <TrackIdBadge trackId={readTrackId(d)} />
      <ClickableHandle type="target" position={Position.Left} id="a" nodeId={id} style={{ top: '25%', transform: 'translateX(-100px)' }} />
      <ClickableHandle type="target" position={Position.Left} id="b" nodeId={id} style={{ top: '75%', transform: 'translateX(-100px)' }} />
      <FlowNodeCard
        family="transformation"
        title={operator}
        content={<span className="text-xs font-black text-slate-100">{operatorSymbol(operator)}</span>}
        subtitle={d.result !== undefined ? `resultado: ${d.result}` : 'esperando entradas'}
      />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} style={{ transform: 'translateX(100px)' }} />
    </div>
  );
}

