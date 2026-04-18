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
};

export function OperatorFlowNode({ id, data }: NodeProps<{ type: 'operator'; data: OperatorFlowNodeData }>) {
  const operator = data?.operator ?? 'adicion';
  const result = data?.result ?? data?.value;
  const showResult = result !== undefined && result !== null;

  return (
    <div className="nopan relative border-2 border-dashed border-cyan-400">
      {/* Debug: muestra el ID del nodo */}
      <div className="absolute -top-5 left-0 text-xs text-cyan-400 bg-black/50 px-1 rounded">
        {id}
      </div>
      {/* Dos entradas: arriba y abajo del borde izquierdo */}
      <ClickableHandle type="target" position={Position.Left} id="a" nodeId={id} style={{ top: '25%' }} />
      <ClickableHandle type="target" position={Position.Left} id="b" nodeId={id} style={{ top: '75%' }} />
      <div className="relative">
        <OperatorCard operator={operator} size="small" />
        {showResult && (
          <div
            className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-sm font-bold text-emerald-600 bg-white/95 px-2 py-0.5 rounded shadow"
            style={{ whiteSpace: 'nowrap' }}
          >
            = {result}
          </div>
        )}
      </div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
