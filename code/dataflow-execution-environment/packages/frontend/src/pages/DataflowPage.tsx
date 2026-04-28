import { useRef, useState, type ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  NumberFlowNode,
  OperatorFlowNode,
  ResultAnchorFlowNode,
  ProgramOutputFlowNode,
  DeckPropFlowNode,
  type ResultViewMode,
} from '@/components/dataflow';
import { NodeProvider, useNode } from '@/contexts/NodeContext';
import { ResultCardUiProvider } from '@/contexts/ResultCardUiContext';
import { SocketInfoFab } from '@/components/SocketInfoFab';
import { getLevelConfig } from '@/data/levelConfig';
import { ModelDeckSidebar } from '../components/ModelDeckSidebar';
import { ArrowLeft, Eye, Volume2 } from 'lucide-react';

const nodeTypes: NodeTypes = {
  number: NumberFlowNode,
  operator: OperatorFlowNode,
  resultAnchor: ResultAnchorFlowNode,
  programOutput: ProgramOutputFlowNode,
  deckProp: DeckPropFlowNode,
};

function speakTitle(title: string, subtitle: string) {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(`${title}. ${subtitle}`);
  u.lang = 'es-ES';
  window.speechSynthesis.speak(u);
}

export function DataflowContent({ isSandbox, levelConfig, backTo, flowContainerRef, showSocketFab = true, rootClassName = 'h-screen w-screen flex flex-col bg-slate-900', extraAsideSections }: {
  isSandbox: boolean;
  levelConfig: ReturnType<typeof getLevelConfig>;
  backTo: string;
  flowContainerRef: React.RefObject<HTMLDivElement | null>;
  showSocketFab?: boolean;
  rootClassName?: string;
  extraAsideSections?: ReactNode;
}) {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    nodesDraggable,
  } = useNode();

  const [viewMode, setViewMode] = useState<ResultViewMode>('abstracto');
  const [showOperatorResults, setShowOperatorResults] = useState(false);

  return (
    <div className={rootClassName}>
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700 shrink-0 gap-4">
        <Link
          to={backTo}
          className="flex items-center gap-3 text-slate-300 hover:text-white transition-colors shrink-0 text-2xl font-semibold px-5 py-4 rounded-xl hover:bg-slate-700/50"
        >
          <ArrowLeft className="w-10 h-10 shrink-0" />
          Volver
        </Link>

        <div className="flex-1 flex flex-col items-center justify-center min-w-0">
          <h1 className="text-2xl font-bold text-white text-center truncate max-w-full">
            {levelConfig.title}
          </h1>
          <p className="text-lg text-slate-400 text-center truncate max-w-full">
            {levelConfig.subtitle}
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
          <button
            type="button"
            onClick={() => setShowOperatorResults((v) => !v)}
            aria-pressed={showOperatorResults}
            className={`flex items-center gap-3 px-5 py-4 min-h-[4.5rem] rounded-xl text-lg font-semibold transition-colors border-2 shadow-lg ${
              showOperatorResults ? 'bg-teal-800 hover:bg-teal-700 border-teal-500 text-teal-50' : 'bg-slate-700 hover:bg-slate-600 border-slate-600 text-slate-100'
            }`}
          >
            <Eye className="w-9 h-9 shrink-0" strokeWidth={2} aria-hidden />
            Mostrar resultados
          </button>
          <div className="mr-4 inline-flex overflow-hidden rounded-xl border-2 border-slate-600 shadow-lg" role="group" aria-label="Modo de visualización">
            <button type="button" onClick={() => setViewMode('concreto')} className={`px-4 py-4 text-xl font-black ${viewMode === 'concreto' ? 'bg-teal-700 text-white' : 'bg-slate-700 text-slate-100'}`}>C</button>
            <button type="button" onClick={() => setViewMode('pictorico')} className={`border-l-2 border-slate-600 px-4 py-4 text-xl font-black ${viewMode === 'pictorico' ? 'bg-teal-700 text-white' : 'bg-slate-700 text-slate-100'}`}>P</button>
            <button type="button" onClick={() => setViewMode('abstracto')} className={`border-l-2 border-slate-600 px-4 py-4 text-xl font-black ${viewMode === 'abstracto' ? 'bg-teal-700 text-white' : 'bg-slate-700 text-slate-100'}`}>A</button>
          </div>
          <button
            type="button"
            onClick={() => speakTitle(levelConfig.title, levelConfig.subtitle)}
            className="p-5 min-h-[4.5rem] min-w-[4.5rem] rounded-xl bg-slate-700 hover:bg-slate-600 text-slate-200 hover:text-white transition-colors border-2 border-slate-600 flex items-center justify-center shadow-lg"
            title="Reproducir título"
          >
            <Volume2 className="w-12 h-12" strokeWidth={2} />
          </button>
        </div>
      </header>

      <div className="flex-1 flex min-h-0">
        {!isSandbox && (
          <aside className="w-64 shrink-0 flex flex-col bg-slate-800 border-r border-slate-700 overflow-y-auto">
            <section className="p-3 border-b border-slate-700">
              <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Mochila
              </h2>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Números (azul), Operadores (rojo, incluye `result`), Figuras (amarillo), Comidas (naranja).
              </p>
              <ModelDeckSidebar />
            </section>
            <section className="p-3 flex-1">
              <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Reglas para el lenguaje
              </h2>
              <p className="text-slate-400 text-base leading-relaxed">
                {levelConfig.rule}
              </p>
            </section>
            {extraAsideSections}
          </aside>
        )}

        {/* Canvas ReactFlow */}
        <div ref={flowContainerRef} className="flex-1 relative min-w-0 bg-black">
          <ResultCardUiProvider viewMode={viewMode} hasExecuted={true}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodeTypes={nodeTypes}
              defaultViewport={{ x: 0, y: 0, zoom: 1 }}
              className="bg-black"
              minZoom={1}
              maxZoom={1}
              zoomOnScroll={false}
              zoomOnPinch={false}
              zoomOnDoubleClick={false}
              panOnDrag={false}
              panOnScroll={false}
              autoPanOnNodeDrag={false}
              nodesDraggable={nodesDraggable}
            >
              <Background color="#334155" gap={16} size={0.5} />
            </ReactFlow>
          </ResultCardUiProvider>
        </div>
      </div>

      {showSocketFab ? <SocketInfoFab /> : null}
    </div>
  );
}

// Main component that provides NodeContext
export default function DataflowPage({ isSandbox }: { isSandbox: boolean }) {
  const params = useParams();
  const worldId = params.worldId;
  const level = params.level;
  const levelConfig = getLevelConfig(worldId, level, isSandbox);
  const backTo = worldId ? (isSandbox ? '/juego' : `/juego/${worldId}`) : '/';

  const flowContainerRef = useRef<HTMLDivElement>(null);

  return (
    <NodeProvider flowContainerRef={flowContainerRef} visionSyncEnabled nodesDraggable={false}>
      <DataflowContent
        isSandbox={isSandbox}
        levelConfig={levelConfig}
        backTo={backTo}
        flowContainerRef={flowContainerRef}
      />
    </NodeProvider>
  );
}
