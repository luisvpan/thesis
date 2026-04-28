import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { OperatorFlowCard } from '@/components/cards/OperatorFlowCard';
import { ClickableHandle } from './ClickableHandle';
import type { OperatorType } from '@/types/card-types';

export type OperatorFlowNodeData = {
  operator: OperatorType;
  result?: number;
  value?: number;
  trackId?: number;
};

export function OperatorFlowNode({ id, data }: NodeProps) {
  const d = (data ?? {}) as OperatorFlowNodeData;
  const operator = d.operator ?? 'adicion';

  return (
    <div className="nopan relative border-2 border-dashed border-red-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      <ClickableHandle type="target" position={Position.Left} id="a" nodeId={id} style={{ top: '25%' }} />
      <ClickableHandle type="target" position={Position.Left} id="b" nodeId={id} style={{ top: '75%' }} />
      <div className="relative">
        <OperatorFlowCard operator={operator} size="small" />
      </div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
