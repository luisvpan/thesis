import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { OperatorCard } from '@/components/cards';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa } from './dataflowResultCpa';
import type { MathOperatorType } from '@/types/card-types';

export type OperatorFlowNodeData = {
  operator: MathOperatorType;
  result?: number;
  /** Valor de salida (igual al resultado) para usar como entrada en otros nodos del dataflow */
  value?: number;
  /** ID de tracking persistente desde YOLO. */
  trackId?: number;
};

type OperatorNode = Node<OperatorFlowNodeData, 'operator'>;

export function OperatorFlowNode({ id, data }: NodeProps<OperatorNode>) {
  const { viewMode, showOperatorResults } = useResultCardUi();
  const operator = data?.operator ?? 'adicion';
  const result = data?.result ?? data?.value;
  const finiteResult =
    typeof result === 'number' && Number.isFinite(result) ? result : null;
  const showBadge = showOperatorResults && finiteResult !== null;

  return (
    <div className="nopan relative border-2 border-dashed border-red-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      {/* Debug: muestra el ID del nodo */}
      <div className="absolute -top-5 left-0 text-xs text-cyan-400 bg-black/50 px-1 rounded">
        {id}
      </div>
      {showBadge && finiteResult !== null ? (
        <div className="absolute bottom-full left-1/2 z-10 mb-2 flex max-w-[min(280px,85vw)] -translate-x-1/2 flex-col items-center gap-1 px-1">
          <div className="rounded-lg border border-slate-600 bg-slate-950/95 px-3 py-2 shadow-xl">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Resultado ({viewMode === 'pictorico' ? 'P' : viewMode === 'concreto' ? 'C' : 'A'})
            </span>
            <div
              className={`mt-1 flex min-h-[2rem] items-center justify-center text-center ${
                viewMode === 'abstracto'
                  ? 'text-2xl font-black tabular-nums text-white'
                  : viewMode === 'concreto'
                    ? 'text-xl font-bold text-sky-300'
                    : 'text-[clamp(0.75rem,2.5vw,1rem)]'
              }`}
            >
              {formatResultCpa(finiteResult, viewMode)}
            </div>
          </div>
        </div>
      ) : null}
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
