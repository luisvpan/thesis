import type { NodeProps } from '@xyflow/react';
import { ShapeCard } from '@/components/cards/ShapeCard';
import { FoodCard } from '@/components/cards/FoodCard';
import type { ShapeType, ShapeSize, ShapeColor } from '@/types/card-types';
import type { FoodType } from '@/types/card-types';

export type DeckPropFlowNodeData =
  | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor; trackId?: number }
  | { variant: 'food'; yoloClass: string; food: FoodType; trackId?: number };

export function DeckPropFlowNode({ data }: NodeProps) {
  const d = data as DeckPropFlowNodeData;
  if (d.variant === 'shape') {
    return (
      <div className="nopan relative rounded-xl border-2 border-yellow-400 bg-slate-900/80 p-2 shadow-lg">
        <ShapeCard shape={d.shape} size={d.size} color={d.color} cardSize="small" isDraggable={false} />
      </div>
    );
  }

  return (
    <div className="nopan relative rounded-xl border-2 border-orange-500 bg-slate-900/80 p-2 shadow-lg">
      <FoodCard food={d.food} size="small" isDraggable={false} />
    </div>
  );
}
