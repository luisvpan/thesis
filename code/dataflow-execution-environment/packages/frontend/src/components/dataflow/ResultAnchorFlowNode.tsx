import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';

/** Carta física detectada como `grapes`: solo marcador visual en el lienzo (no forma parte del lenguaje). */
export type ResultAnchorFlowNodeData = Record<string, never>;

export function ResultAnchorFlowNode({ id }: NodeProps) {
  return (
    <div className="nopan relative border-2 border-dashed border-violet-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      <div className="absolute -top-5 left-0 text-xs text-violet-300 bg-black/50 px-1 rounded">
        {id}
      </div>
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />
      <div className="flex h-full flex-col items-center justify-center px-3">
        <p className="text-center text-lg font-bold text-violet-200 leading-tight">Marcador</p>
        <p className="mt-2 text-center text-sm text-slate-400 leading-snug">
          Referencia física para la carta de resultado (no muestra valores del programa).
        </p>
      </div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
