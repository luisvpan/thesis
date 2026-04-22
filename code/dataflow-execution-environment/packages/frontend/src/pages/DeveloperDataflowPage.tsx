import { useEffect, useRef, useState } from 'react';
import { NodeProvider, useNode } from '@/contexts/NodeContext';
import { DataflowContent } from '@/pages/DataflowPage';
import { DEV_TOOLBOX_CONFIG } from '@/data/levelConfig';
import {
  ensureDataflowExecuteSocket,
  executeProgramViaWs,
  subscribeDataflowWsStatus,
} from '@/services/dataflowExecuteWs';

function ResultAnchorAside() {
  const { addResultAnchorPair } = useNode();

  return (
    <section className="p-3 border-t border-slate-700 shrink-0">
      <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
        Marcador físico (visión)
      </h2>
      <button
        type="button"
        onClick={addResultAnchorPair}
        className="w-full rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-base py-3 px-4 transition-colors shadow-lg text-center"
      >
        Grapes + carta resultado
      </button>
      <p className="mt-2 text-xs text-slate-500 leading-relaxed">
        Equivalente a la etiqueta «grapes» que envía el detector: ancla la salida junto a la carta de
        resultado.
      </p>
    </section>
  );
}

function WsExecuteStatus() {
  const [connected, setConnected] = useState(false);
  useEffect(() => subscribeDataflowWsStatus(setConnected), []);
  useEffect(() => {
    void ensureDataflowExecuteSocket().catch(() => {
      /* primer intento de conexión para el badge */
    });
  }, []);

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
        connected ? 'bg-emerald-900/80 text-emerald-300' : 'bg-rose-900/70 text-rose-200'
      }`}
    >
      <span className="h-2 w-2 rounded-full bg-current opacity-90" aria-hidden />
      WS ejecución: {connected ? 'conectado' : 'desconectado'}
    </span>
  );
}

export default function DeveloperDataflowPage() {
  const flowContainerRef = useRef<HTMLDivElement>(null);

  return (
    <NodeProvider
      flowContainerRef={flowContainerRef}
      visionSyncEnabled={false}
      executeRunner={executeProgramViaWs}
    >
      <div className="h-screen w-screen flex flex-col bg-slate-900">
        <header className="flex items-center gap-4 px-4 py-2.5 bg-slate-950 border-b border-slate-700 shrink-0">
          <WsExecuteStatus />
          <p className="text-sm text-slate-400 flex-1 min-w-0">
            Resultados automáticos al editar el grafo vía{' '}
            <code className="text-slate-300">/ws/dataflow</code> (compilador + runtime). Misma mochila
            que las etiquetas YOLO: 0–9, + − × ÷ y grapes.
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
            extraAsideSections={<ResultAnchorAside />}
          />
        </div>
      </div>
    </NodeProvider>
  );
}
