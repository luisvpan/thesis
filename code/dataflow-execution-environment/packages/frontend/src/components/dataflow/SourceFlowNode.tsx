import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { ShapeType, ShapeSize, ShapeColor, FoodType } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';

export type SourceFlowNodeData =
  | { variant: 'number'; value: number; visionSubtitle?: string; trackId?: number }
  | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor; trackId?: number }
  | { variant: 'food'; yoloClass: string; food: FoodType; trackId?: number };

export type SourceFlowNode = Node<SourceFlowNodeData, 'source'>;

function sourceTitle(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return 'Numero';
  if (data.variant === 'shape') return 'Forma';
  return 'Comida';
}

function sourceMain(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return String(data.value);
  if (data.variant === 'shape') return `${data.shape} ${data.size}`;
  return data.food;
}

export function SourceFlowNode({ id, data }: NodeProps<SourceFlowNode>) {
  const d = (data ?? { variant: 'number', value: 0 }) as SourceFlowNodeData;
  const subtitle = d.variant === 'number' ? d.visionSubtitle : undefined;

  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[25%]">
      <FlowNodeCard family="input" title={sourceTitle(d)} content={<span className="text-xs font-black text-slate-100">{sourceMain(d)}</span>} subtitle={subtitle} />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
