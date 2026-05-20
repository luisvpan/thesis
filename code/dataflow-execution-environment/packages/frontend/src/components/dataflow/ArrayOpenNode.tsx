import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { ClickableHandle } from './ClickableHandle';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';

export type ArrayOpenNodeData = VisionNodeMeta;
export type ArrayOpenNode = Node<ArrayOpenNodeData, 'arrayOpen'>;

export function ArrayOpenNode({ id, data }: NodeProps<ArrayOpenNode>) {
  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[45%]">
      <TrackIdBadge trackId={readTrackId(data)} />
      <FlowNodeCard
        family="input"
        title="Abrir arreglo"
        content={<span className="text-xs font-black text-slate-100">[</span>}
      />
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="zone-out"
        nodeId={id}
        style={{ transform: 'translateX(100px)' }}
      />
    </div>
  );
}
