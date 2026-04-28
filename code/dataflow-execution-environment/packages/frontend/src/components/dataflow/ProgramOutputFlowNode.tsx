import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa } from './dataflowResultCpa';

/** Solo frontend: muestra salida tras ejecutar; valor numérico para el grafo oculto al backend como DataSource. */
export type ProgramOutputFlowNodeData = {
  /** Copia del último resultado ejecutado para propagación local del grafo */
  value?: number;
};

export function ProgramOutputFlowNode({
  id,
}: NodeProps) {
  const { executionResult, executionError } = useNode();
  const { viewMode, hasExecuted } = useResultCardUi();

  const display =
    executionError ? (
      <p className="text-lg font-semibold text-red-400 text-center leading-snug px-1">{executionError}</p>
    ) : hasExecuted && executionResult !== null ? (
      <div className="flex flex-col items-center gap-1 text-white">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Resultado ({viewMode === 'pictorico' ? 'P' : viewMode === 'concreto' ? 'C' : 'A'})
        </span>
        <div
          className={
            viewMode === 'abstracto'
              ? 'text-5xl font-black text-white tabular-nums drop-shadow-lg'
              : viewMode === 'concreto'
                ? 'text-3xl font-bold text-sky-300 text-center drop-shadow-md'
                : 'text-center drop-shadow-md max-w-[14rem]'
          }
        >
          {formatResultCpa(executionResult, viewMode)}
        </div>
      </div>
    ) : (
      <p className="text-base text-slate-500 text-center italic px-2">Ejecutá para ver el resultado aquí</p>
    );

  return (
    <div className="nopan relative border-2 border-dashed border-teal-400 w-60 h-60 -translate-y-[25%] -translate-x-[30%]">
      <div className="absolute -top-5 left-0 text-xs text-teal-300 bg-black/50 px-1 rounded">
        {id}
      </div>
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />
      <div className="flex h-full flex-col items-center justify-center px-2 py-1">{display}</div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
