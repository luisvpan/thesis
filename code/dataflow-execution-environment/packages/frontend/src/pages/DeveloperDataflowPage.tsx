import { useRef } from 'react';
import { NodeProvider, useNode } from '@/contexts/NodeContext';
import { DataflowContent } from '@/pages/DataflowPage';
import { DEV_TOOLBOX_CONFIG } from '@/data/levelConfig';
import { executeProgram } from '@/services/executeProgram';

function ResultToolsAside() {
  const { addResultCard, addResultAnchorPair } = useNode();

  return (
    <section className="p-3 border-t border-slate-700 shrink-0 space-y-4">
      <div>
        <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Carta resultado
        </h2>
        <button
          type="button"
          onClick={addResultCard}
          className="w-full rounded-xl bg-emerald-700 hover:bg-emerald-600 text-white font-bold text-base py-3 px-4 transition-colors shadow-lg text-center"
        >
          Añadir carta resultado
        </button>
        <p className="mt-2 text-xs text-slate-500 leading-relaxed">
          Una sola carta: conectá la salida del operador al puerto izquierdo. El número se calcula en el
          navegador.
        </p>
      </div>
      <div>
        <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Par uva (visión física)
        </h2>
        <button
          type="button"
          onClick={addResultAnchorPair}
          className="w-full rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-base py-3 px-4 transition-colors shadow-lg text-center"
        >
          Marcador + carta (uva)
        </button>
        <p className="mt-2 text-xs text-slate-500 leading-relaxed">
          Equivalente a la detección «grapes»: marcador y mitad resultado como en la mesa física.
        </p>
      </div>
    </section>
  );
}

export default function DeveloperDataflowPage() {
  const flowContainerRef = useRef<HTMLDivElement>(null);

  return (
    <NodeProvider
      flowContainerRef={flowContainerRef}
      visionSyncEnabled={false}
      executeRunner={executeProgram}
      nodesDraggable
    >
      <div className="h-screen w-screen flex flex-col bg-slate-900">
        <header className="flex items-center gap-4 px-4 py-2.5 bg-slate-950 border-b border-slate-700 shrink-0">
          <span className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide bg-emerald-900/80 text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-current opacity-90" aria-hidden />
            Intérprete local (navegador)
          </span>
          <p className="text-sm text-slate-400 flex-1 min-w-0">
            Sin WebSocket de ejecución: el resultado sale del intérprete dataflow en el navegador.
            Mochila completa YOLO (32 clases); en este modo podés arrastrar nodos en el lienzo.
          </p>
        </header>
        <div className="flex-1 flex flex-col min-h-0 min-w-0">
          <DataflowContent
            isSandbox={false}
            levelConfig={DEV_TOOLBOX_CONFIG}
            backTo="/"
            flowContainerRef={flowContainerRef}
            showSocketFab={false}
            rootClassName="flex-1 flex flex-col min-h-0 min-w-0 bg-slate-900"
            extraAsideSections={<ResultToolsAside />}
          />
        </div>
      </div>
    </NodeProvider>
  );
}
