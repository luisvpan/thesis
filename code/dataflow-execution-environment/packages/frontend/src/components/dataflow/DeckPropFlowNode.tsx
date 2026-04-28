import type { NodeProps } from '@xyflow/react';
import { ShapeCard } from '@/components/cards/ShapeCard';
import { FoodCard } from '@/components/cards/FoodCard';
import type { ShapeType, ShapeSize, ShapeColor } from '@/types/card-types';
import type { FoodType } from '@/types/card-types';

export type DeckPropFlowNodeData =
  | {
      variant: 'shape';
      yoloClass: string;
      shape: ShapeType;
      size: ShapeSize;
      color: ShapeColor;
      trackId?: number;
    }
  | {
      variant: 'food';
      yoloClass: string;
      food: FoodType;
      trackId?: number;
    };

type DeckPropNode = NodeProps<{ type: 'deckProp'; data: DeckPropFlowNodeData }>;

export function DeckPropFlowNode({ data }: DeckPropNode) {
  if (data.variant === 'shape') {
    return (
      <div className="nopan relative rounded-xl border-2 border-yellow-400 bg-slate-900/80 p-2 shadow-lg">
        <ShapeCard
          shape={data.shape}
          size={data.size}
          color={data.color}
          cardSize="small"
          isDraggable={false}
        />
      </div>
    );
  }

  return (
    <div className="nopan relative rounded-xl border-2 border-orange-500 bg-slate-900/80 p-2 shadow-lg">
      <FoodCard food={data.food} size="small" isDraggable={false} />
    </div>
  );
}
