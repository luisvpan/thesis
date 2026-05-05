import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';

export type ArrayCloseNodeData = Record<string, never>;
export type ArrayCloseNode = Node<ArrayCloseNodeData, 'arrayClose'>;

export function ArrayCloseNode({ id }: NodeProps<ArrayCloseNode>) {
  return (
    <div className="relative flex items-center justify-center -translate-x-[30%] -translate-y-[25%]">
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="zone-in"
        nodeId={id}
      />
      <div className="flex items-center justify-center w-16 h-20 rounded-r-2xl border-4 border-l-0 border-teal-400 bg-teal-950/70 shadow-lg shadow-teal-900/40">
        <span className="text-5xl font-black text-teal-300 select-none leading-none font-mono">]</span>
      </div>
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
      />
    </div>
  );
}
