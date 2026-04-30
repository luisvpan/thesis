import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa } from './dataflowResultCpa';
import { FlowNodeCard } from './FlowNodeCard';

/** Solo frontend: muestra salida tras ejecutar; valor numérico o descripción semántica. */
export type ProgramOutputFlowNodeData = {
  /** Valor numérico para resultados racionales */
  value?: number;
  /** Descripción semántica para resultados de arreglo */
  description?: string;
};

export type ProgramOutputFlowNode = Node<ProgramOutputFlowNodeData, 'programOutput'>;

export function ProgramOutputFlowNode({
  id,
  data,
}: NodeProps<ProgramOutputFlowNode>) {
  const { executionError } = useNode();
  const { viewMode } = useResultCardUi();

  const value = data.value;
  const description = data.description;

  const display =
    executionError ? (
      <p className="text-lg font-semibold text-red-400 text-center leading-snug px-1">{executionError}</p>
    ) : description ? (
      // Resultado semántico (arreglo de objetos)
      <div className="flex flex-col items-center gap-1 text-white">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          Resultado
        </span>
        <p className="text-lg text-center text-teal-200 leading-snug px-1">
          {description}
        </p>
      </div>
    ) : value !== undefined ? (
      // Resultado numérico
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
          {formatResultCpa(value, viewMode)}
        </div>
      </div>
    ) : (
      <p className="text-base text-slate-500 text-center italic px-2">Ejecuta para ver el resultado aqui</p>
    );

  return (
    <div className="relative h-52 w-52 -translate-x-[30%] -translate-y-[25%]">
      <div className="absolute -top-5 left-0 rounded bg-black/50 px-1 text-xs text-teal-300">
        {id}
      </div>
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} />
      <FlowNodeCard family="sink" title="Salida" content={display} />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} />
    </div>
  );
}
