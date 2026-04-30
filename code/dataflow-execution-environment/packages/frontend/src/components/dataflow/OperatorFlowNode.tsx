import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { OperatorType } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';

export type OperatorFlowNodeData = {
  operator: OperatorType;
  result?: number;
  value?: number;
  trackId?: number;
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
      <ClickableHandle type="target" position={Position.Left} id="a" nodeId={id} style={{ top: '25%' }} />
      <ClickableHandle type="target" position={Position.Left} id="b" nodeId={id} style={{ top: '75%' }} />
      <FlowNodeCard
        family="transformation"
        title={operator}
        content={<span className="text-4xl font-black text-slate-100">{operatorSymbol(operator)}</span>}
        subtitle={d.result !== undefined ? `resultado: ${d.result}` : 'esperando entradas'}
      />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
