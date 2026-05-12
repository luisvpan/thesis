import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ARRAY_CLOSE_ZONE_IN_TOP_FRAC } from '../../utils/arrayZoneGeometry';
import { ClickableHandle } from './ClickableHandle';
import { FlowNodeCard } from './FlowNodeCard';

export type ArrayCloseNodeData = Record<string, never>;
export type ArrayCloseNode = Node<ArrayCloseNodeData, 'arrayClose'>;

export function ArrayCloseNode({ id }: NodeProps<ArrayCloseNode>) {
  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[25%]">
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="zone-in"
        nodeId={id}
        style={{
          top: `${ARRAY_CLOSE_ZONE_IN_TOP_FRAC * 100}%`,
          transform: 'translateX(-100px)',
        }}
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
          top: '72%',
          transform: 'translateX(100px)',
        }}
      />
    </div>
  );
}
