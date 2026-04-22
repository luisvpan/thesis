import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';

/** Carta física «uvas» / `grapes`: marcador; el valor sale del operador que alimenta este nodo. */
export type ResultAnchorFlowNodeData = {
  /** Id de la carta de resultado emparejada (arista marcador.out → carta.in). */
  pairedOutputId?: string;
};

export function ResultAnchorFlowNode({ id }: NodeProps<{ type: 'resultAnchor'; data: ResultAnchorFlowNodeData }>) {
  return (
    <div className="nopan relative border-2 border-dashed border-violet-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      <div className="absolute -top-5 left-0 text-xs text-violet-300 bg-black/50 px-1 rounded">
        {id}
      </div>
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />
      <div className="flex h-full flex-col items-center justify-center px-3">
        <p className="text-center text-lg font-bold text-violet-200 leading-tight">Marcador</p>
        <p className="mt-2 text-center text-sm text-slate-400 leading-snug">
          Conectá la salida de un operador aquí; cada par uva tiene su propio resultado en la carta
          derecha.
        </p>
      </div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
