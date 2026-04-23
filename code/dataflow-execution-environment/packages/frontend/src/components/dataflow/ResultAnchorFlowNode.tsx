import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import {
  UVA_LEFT_PANEL_PX,
  UVA_PAIR_TRANSFORM_CLASS,
} from '@/utils/uvaCardLayout';
import { ClickableHandle } from './ClickableHandle';

/** Carta física «uvas» / `grapes`: marcador; el valor sale del operador que alimenta este nodo. */
export type ResultAnchorFlowNodeData = {
  /** Id de la carta de resultado emparejada (arista marcador.out → carta.in). */
  pairedOutputId?: string;
};

/** Panel izquierdo de la carta uva fusionada (mismo ancho que una carta estándar). */
export function ResultAnchorFlowNode({ id }: NodeProps<{ type: 'resultAnchor'; data: ResultAnchorFlowNodeData }>) {
  return (
    <div
      className={`nopan relative flex h-60 flex-col items-center justify-center rounded-l-xl border-y-2 border-l-2 border-r-0 border-dashed border-violet-400 bg-slate-950/95 px-2 shadow-lg ${UVA_PAIR_TRANSFORM_CLASS}`}
      style={{ width: UVA_LEFT_PANEL_PX }}
    >
      <div className="pointer-events-none absolute -top-5 left-0 max-w-[280px] truncate text-xs text-violet-300 bg-black/50 px-1 rounded">
        {id}
      </div>
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />

    </div>
  );
}
