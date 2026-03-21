import type { NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import { NumberCard } from '@/components/cards';

export type NumberFlowNodeData = {
  value: number;
};

export function NumberFlowNode({ data }: NodeProps<{ type: 'number'; data: NumberFlowNodeData }>) {
  const value = data?.value ?? 0;
  return (
    <div className="nopan">
      <Handle type="target" position={Position.Left} id="in" className="!w-3 !h-3 !border-2 !bg-white" />
      <NumberCard value={value} size="small" />
      <Handle type="source" position={Position.Right} id="out" className="!w-3 !h-3 !border-2 !bg-white" />
    </div>
  );
}
