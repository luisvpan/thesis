import { useEffect } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { ShapeType, ShapeSize, ShapeColor, FoodType, MontessoriColor, CapColor, StickColor } from '@/types/card-types';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { useNode } from '@/contexts/NodeContext';
import type { HandleKind } from './handle-kinds';
import { SOURCE_NODE_WRAPPER_CLASS } from './source-flow/sourceNodeLayout';
import { renderSourceFlowNodeBody } from './source-flow/renderSourceFlowNodeBody';

type VisionSynced = VisionNodeMeta;

export type SourceFlowNodeData = VisionSynced &
  (
    | { variant: 'number'; value: number; visionSubtitle?: string }
    | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
    | { variant: 'food'; yoloClass: string; food: FoodType }
    | { variant: 'montessori'; yoloClass: string; color: MontessoriColor }
    | { variant: 'cap'; yoloClass: string; color: CapColor }
    | { variant: 'stick'; yoloClass: string; color: StickColor }
  );

export type SourceFlowNode = Node<SourceFlowNodeData, 'source'>;

export function SourceFlowNode({ id, data }: NodeProps<SourceFlowNode>) {
  const d = (data ?? { variant: 'number', value: 0 }) as SourceFlowNodeData;
  const { registerPortKind, unregisterPortKinds } = useNode();

  const produces: HandleKind = d.variant === 'number' ? 'rational' : 'cpa';
  const wrapperClass = SOURCE_NODE_WRAPPER_CLASS[d.variant];

  useEffect(() => {
    registerPortKind(id, 'out', { produces });
    return () => unregisterPortKinds(id);
  }, [id, produces, registerPortKind, unregisterPortKinds]);

  return (
    <div className={wrapperClass}>
      <TrackIdBadge trackId={readTrackId(d)} />
      {renderSourceFlowNodeBody(d)}
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
        produces={produces}
        style={{ transform: 'translateX(100px)' }}
      />
    </div>
  );
}
