import { useEffect } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { useNode } from '@/contexts/NodeContext';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import {
  ARRAY_CLOSE_ZONE_IN_TOP_FRAC,
  ARRAY_ZONE_IN_HANDLE_STYLE,
} from './arrayNodeLayout';
import { ClickableHandle } from './ClickableHandle';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';

export type ArrayCloseNodeData = VisionNodeMeta;
export type ArrayCloseNode = Node<ArrayCloseNodeData, 'arrayClose'>;

export function ArrayCloseNode({ id, data }: NodeProps<ArrayCloseNode>) {
  const { registerPortKind, unregisterPortKinds } = useNode();
  const shellClass = useFlowNodeShellClass();

  useEffect(() => {
    registerPortKind(id, 'zone-in', { accepts: ['any'] });
    registerPortKind(id, 'out', { produces: 'any' });
    return () => unregisterPortKinds(id);
  }, [id, registerPortKind, unregisterPortKinds]);

  return (
    <div className={`relative h-52 w-26 -translate-x-[10%] -translate-y-[45%] ${shellClass}`}>
      <TrackIdBadge trackId={readTrackId(data)} />
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="zone-in"
        nodeId={id}
        handleVariant="zone-close-triangle"
        accepts={['any']}
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
        handleVariant="zone-close-out"
        produces="any"
        style={{
          top: `${ARRAY_CLOSE_ZONE_IN_TOP_FRAC * 100}%`,
          transform: 'translateX(100px)',
        }}
      />
    </div>
  );
}
