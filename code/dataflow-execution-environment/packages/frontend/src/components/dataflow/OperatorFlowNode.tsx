import type { NodeProps } from '@xyflow/react';
import { Handle, Position } from '@xyflow/react';
import { OperatorCard } from '@/components/cards';
import type { MathOperatorType } from '@/types/card-types';

export type OperatorFlowNodeData = {
  operator: MathOperatorType;
  result?: number;
  /** Valor de salida (igual al resultado) para usar como entrada en otros nodos del dataflow */
  value?: number;
};

export function OperatorFlowNode({ data }: NodeProps<{ type: 'operator'; data: OperatorFlowNodeData }>) {
  const operator = data?.operator ?? 'adicion';
  const result = data?.result ?? data?.value;
  const showResult = result !== undefined && result !== null;

  return (
    <div className="nopan relative">
      {/* Dos entradas: arriba y abajo del borde izquierdo */}
      <Handle type="target" position={Position.Left} id="a" className="!w-3 !h-3 !border-2 !bg-white !top-1/4" style={{ top: '25%' }} />
      <Handle type="target" position={Position.Left} id="b" className="!w-3 !h-3 !border-2 !bg-white !top-3/4" style={{ top: '75%' }} />
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
      <Handle type="source" position={Position.Right} id="out" className="!w-3 !h-3 !border-2 !bg-white" />
    </div>
  );
}
