import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import { FlowNodeCard } from './FlowNodeCard';

export type ArrayOpenNodeData = Record<string, never>;
export type ArrayOpenNode = Node<ArrayOpenNodeData, 'arrayOpen'>;

export function ArrayOpenNode({ id }: NodeProps<ArrayOpenNode>) {
  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[45%]">
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
