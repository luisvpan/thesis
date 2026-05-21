import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import {
  ARRAY_CLOSE_ZONE_IN_TOP_FRAC,
  ARRAY_ZONE_IN_HANDLE_STYLE,
} from './arrayNodeLayout';
import { ClickableHandle } from './ClickableHandle';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';

export type ArrayCloseNodeData = VisionNodeMeta;
export type ArrayCloseNode = Node<ArrayCloseNodeData, 'arrayClose'>;

export function ArrayCloseNode({ id, data }: NodeProps<ArrayCloseNode>) {
  return (
    <div className="relative h-52 w-26 -translate-x-[10%] -translate-y-[45%]">
      <TrackIdBadge trackId={readTrackId(data)} />
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="zone-in"
        nodeId={id}
        style={ARRAY_ZONE_IN_HANDLE_STYLE}
      />
      <FlowNodeCard
        family="transformation"
        title="Cerrar arreglo"
        content={<span className="text-xs font-black text-slate-100">]</span>}
      />
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
        style={{
          top: `${ARRAY_CLOSE_ZONE_IN_TOP_FRAC * 100}%`,
          transform: 'translateX(100px)',
        }}
      />
    </div>
  );
}
