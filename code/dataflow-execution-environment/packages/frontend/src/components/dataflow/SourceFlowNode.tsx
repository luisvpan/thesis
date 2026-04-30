import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { ShapeType, ShapeSize, ShapeColor, FoodType } from '@/types/card-types';

export type SourceFlowNodeData =
  | { variant: 'number'; value: number; visionSubtitle?: string; trackId?: number }
  | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor; trackId?: number }
  | { variant: 'food'; yoloClass: string; food: FoodType; trackId?: number };

function getBorderColor(variant: string): string {
  switch (variant) {
    case 'number': return 'border-blue-400';
    case 'shape': return 'border-yellow-400';
    case 'food': return 'border-orange-500';
    default: return 'border-gray-400';
  }
}

export function SourceFlowNode({ id, data }: NodeProps) {
  const d = (data ?? { variant: 'number', value: 0 }) as SourceFlowNodeData;
  const borderColor = getBorderColor(d.variant);

  return (
    <div className={`nopan relative border-2 border-dashed ${borderColor} w-60 h-60 -translate-y-[25%] -translate-x-[30%]`}>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
