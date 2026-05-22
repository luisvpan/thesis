import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { ARRAY_ZONE_OUT_HANDLE_STYLE } from './arrayNodeLayout';
import { ClickableHandle } from './ClickableHandle';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';

export type ArrayOpenNodeData = VisionNodeMeta;
export type ArrayOpenNode = Node<ArrayOpenNodeData, 'arrayOpen'>;

export function ArrayOpenNode({ id, data }: NodeProps<ArrayOpenNode>) {
  const shellClass = useFlowNodeShellClass();
  return (
    <div className={`relative h-52 w-26 -translate-x-[10%] -translate-y-[45%] ${shellClass}`}>
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
        style={ARRAY_ZONE_OUT_HANDLE_STYLE}
      />
    </div>
  );
}
