import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { OperatorCard } from '@/components/cards';
import { ClickableHandle } from './ClickableHandle';
import type { MathOperatorType } from '@/types/card-types';

export type OperatorFlowNodeData = {
  operator: MathOperatorType;
  result?: number;
  /** Valor de salida (igual al resultado) para usar como entrada en otros nodos del dataflow */
  value?: number;
  /** ID de tracking persistente desde YOLO. */
  trackId?: number;
};

export function OperatorFlowNode({ id, data }: NodeProps<{ type: 'operator'; data: OperatorFlowNodeData }>) {
  const operator = data?.operator ?? 'adicion';
  console.log("OYE, PASAME ESE DATICOOOO => ", data)
  const result = data?.result ?? data?.value;
  console.log("RESULTADOOO =>", result)
  const showResult = result !== undefined && result !== null;

  return (
    <div className="nopan relative border-2 border-dashed border-red-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      {/* Debug: muestra el ID del nodo */}
      <div className="absolute -top-5 left-0 text-xs text-cyan-400 bg-black/50 px-1 rounded">
        {id}
      </div>
      {/* Dos entradas: arriba y abajo del borde izquierdo */}
      <ClickableHandle type="target" position={Position.Left} id="a" nodeId={id} style={{ top: '25%' }} />
      <ClickableHandle type="target" position={Position.Left} id="b" nodeId={id} style={{ top: '75%' }} />
      <div className="relative">
        <OperatorCard operator={operator} size="small" />
      </div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
