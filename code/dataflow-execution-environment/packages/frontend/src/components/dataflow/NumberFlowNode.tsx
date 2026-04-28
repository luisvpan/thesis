import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { NumberCard } from '@/components/cards';
import { ClickableHandle } from './ClickableHandle';

export type NumberFlowNodeData = {
  value: number;
  visionSubtitle?: string;
  trackId?: number;
};

export function NumberFlowNode({ id, data }: NodeProps) {
  const d = (data ?? {}) as NumberFlowNodeData;
  const value = d.value ?? 0;
  return (
    <div className="nopan relative border-2 border-dashed border-blue-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />
      <NumberCard value={value} subtitle={d.visionSubtitle} size="small" />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
