import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';

export type ArrayOpenNodeData = Record<string, never>;
export type ArrayOpenNode = Node<ArrayOpenNodeData, 'arrayOpen'>;

export function ArrayOpenNode({ id }: NodeProps<ArrayOpenNode>) {
  return (
    <div className="relative flex items-center justify-center -translate-x-[30%] -translate-y-[25%]">
      <div className="flex items-center justify-center w-16 h-20 rounded-l-2xl border-4 border-r-0 border-teal-400 bg-teal-950/70 shadow-lg shadow-teal-900/40">
        <span className="text-5xl font-black text-teal-300 select-none leading-none font-mono">[</span>
      </div>
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="zone-out"
        nodeId={id}
      />
    </div>
  );
}
