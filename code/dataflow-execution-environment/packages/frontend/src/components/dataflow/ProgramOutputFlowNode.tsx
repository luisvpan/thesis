import type { NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import {
  UVA_NORMAL_CARD_PX,
  UVA_PAIR_TRANSFORM_CLASS,
  UVA_RIGHT_PANEL_PX,
} from '@/utils/uvaCardLayout';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa } from './dataflowResultCpa';

/** Solo frontend: resultado local hasta el operador que alimenta esta carta. */
export type ProgramOutputFlowNodeData = {
  value?: number;
  tapError?: string;
  /** Id del `resultAnchor` emparejado (marcador físico / par uva). */
  pairedAnchorId?: string;
  /**
   * `uvaRight`: mitad derecha del par uva (marcador + resultado).
   * `result`: una sola carta “Resultado”; conectá `operator.out` → `in`.
   */
  displayMode?: 'uvaRight' | 'result';
};

export function ProgramOutputFlowNode({
  id,
  data,
}: NodeProps<{ type: 'programOutput'; data: ProgramOutputFlowNodeData }>) {
  const { executionResult, executionError, isExecuting } = useNode();
  const { viewMode } = useResultCardUi();

  const tapError = data?.tapError;
  const localVal = data?.value;
  const displayMode = data?.displayMode ?? 'uvaRight';
  const hasPair = Boolean(data?.pairedAnchorId);
  const showNumeric =
    typeof localVal === "number" && !Number.isNaN(localVal);

  const display =
    tapError ? (
      <p className="text-lg font-semibold text-red-400 text-center leading-snug px-1">{tapError}</p>
    ) : showNumeric ? (
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
          {formatResultCpa(localVal, viewMode)}
        </div>
      </div>
    ) : hasPair && !tapError ? (
      <p className="text-base text-slate-500 text-center italic px-2">
        {isExecuting
          ? 'Actualizando resultado…'
          : displayMode === 'result'
            ? 'Conectá la salida del operador a la entrada izquierda de esta carta.'
            : 'Sin resultado en este punto (revisá el operador conectado al marcador).'}
      </p>
    ) : !hasPair &&
      executionResult !== null &&
      executionResult !== undefined ? (
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
    ) : executionError && !hasPair ? (
      <p className="text-lg font-semibold text-red-400 text-center leading-snug px-1">{executionError}</p>
    ) : isExecuting ? (
      <p className="text-base text-sky-400 text-center italic px-2">Actualizando resultado…</p>
    ) : (
      <p className="text-base text-slate-500 text-center italic px-2">
        {displayMode === 'result'
          ? 'Conectá la salida del operador al puerto izquierdo de esta carta.'
          : 'El resultado aparece aquí cuando el flujo envía datos.'}
      </p>
    );

  const isResultCard = displayMode === 'result';

  return (
    <div
      className={
        isResultCard
          ? 'nopan relative flex h-60 w-60 flex-col items-center justify-center rounded-2xl border-2 border-dashed border-emerald-400 bg-slate-950/95 px-3 shadow-lg'
          : `nopan relative flex h-60 flex-col items-center justify-center rounded-r-xl border-y-2 border-r-2 border-l border-dashed border-teal-400 border-l-slate-600 bg-slate-950/95 px-2 shadow-lg ${UVA_PAIR_TRANSFORM_CLASS}`
      }
      style={isResultCard ? { width: UVA_NORMAL_CARD_PX } : { width: UVA_RIGHT_PANEL_PX }}
    >
      <div
        className={`pointer-events-none absolute -top-5 max-w-full truncate text-[10px] bg-black/40 px-1 rounded ${
          isResultCard
            ? 'left-0 text-emerald-300'
            : 'right-0 text-right text-teal-300/90'
        }`}
      >
        {isResultCard ? `Resultado · ${id}` : id}
      </div>
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />
      <div className="flex w-full flex-col items-center justify-center px-1 py-1">
        {display}
      </div>
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
