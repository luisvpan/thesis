import { useCallback, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { NumberFlowNode, OperatorFlowNode } from '@/components/dataflow';
import { NodeProvider, useNode } from '@/contexts/NodeContext';
import { VisionDetectedBadge } from '@/components/VisionDetectedBadge';
import { getLevelConfig } from '@/data/levelConfig';
import { ArrowLeft, Plus, Minus, Volume2, Play } from 'lucide-react';

type ViewMode = 'pictorico' | 'concreto' | 'abstracto';

const VIEW_MODE_LABELS: Record<ViewMode, string> = {
  pictorico: 'Pictórico',
  concreto: 'Concreto',
  abstracto: 'Abstracto',
};

const nodeTypes: NodeTypes = {
  number: NumberFlowNode,
  operator: OperatorFlowNode,
};

function speakTitle(title: string, subtitle: string) {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(`${title}. ${subtitle}`);
  u.lang = 'es-ES';
  window.speechSynthesis.speak(u);
}

// Inner component that uses NodeContext
function DataflowContent({ isSandbox, levelConfig, backTo, flowContainerRef }: {
  isSandbox: boolean;
  levelConfig: ReturnType<typeof getLevelConfig>;
  backTo: string;
  flowContainerRef: React.RefObject<HTMLDivElement | null>;
}) {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    addNumberNode,
    addOperatorNode,
    getExecutionResult,
  } = useNode();

  const [viewMode, setViewMode] = useState<ViewMode>('pictorico');
  const [executedResult, setExecutedResult] = useState<number | null>(null);

  const cycleViewMode = useCallback(() => {
    setViewMode((m) => (m === 'pictorico' ? 'concreto' : m === 'concreto' ? 'abstracto' : 'pictorico'));
  }, []);

  const onExecute = useCallback(() => {
    setExecutedResult(getExecutionResult());
  }, [getExecutionResult]);

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-900">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-slate-800 border-b border-slate-700 shrink-0 gap-4">
        <Link
          to={backTo}
          className="flex items-center gap-2 text-slate-300 hover:text-white transition-colors shrink-0 text-lg"
        >
          <ArrowLeft className="w-5 h-5" />
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
          <VisionDetectedBadge />
          <button
            type="button"
            onClick={onExecute}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-teal-500 hover:bg-teal-600 text-white text-lg font-medium transition-colors"
            title="Ejecutar"
          >
            <Play className="w-4 h-4" />
            Ejecutar
          </button>
          <button
            type="button"
            onClick={cycleViewMode}
            className="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-lg font-medium transition-colors border border-slate-600"
          >
            {VIEW_MODE_LABELS[viewMode]}
          </button>
          <button
            type="button"
            onClick={() => speakTitle(levelConfig.title, levelConfig.subtitle)}
            className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-300 hover:text-white transition-colors border border-slate-600"
            title="Reproducir título"
          >
            <Volume2 className="w-5 h-5" />
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
              <div className="space-y-3">
                <div>
                  <p className="text-base font-medium text-slate-400 mb-2">Añadir número</p>
                  <div className="flex flex-wrap gap-1">
                    {levelConfig.numbers.map((n) => (
                      <button
                        key={n}
                        type="button"
                        onClick={() => addNumberNode(n)}
                        className="w-9 h-9 rounded-lg bg-slate-700 hover:bg-teal-500 text-slate-200 hover:text-white font-bold text-lg transition-colors"
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-base font-medium text-slate-400 mb-2">Añadir operador</p>
                  <div className="flex gap-2">
                    {levelConfig.operators.includes('adicion') && (
                      <button
                        type="button"
                        onClick={() => addOperatorNode('adicion')}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-teal-500 hover:bg-teal-600 text-white font-bold text-lg transition-colors"
                      >
                        <Plus className="w-4 h-4" />
                        Suma
                      </button>
                    )}
                    {levelConfig.operators.includes('sustraccion') && (
                      <button
                        type="button"
                        onClick={() => addOperatorNode('sustraccion')}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-rose-500 hover:bg-rose-600 text-white font-bold text-lg transition-colors"
                      >
                        <Minus className="w-4 h-4" />
                        Resta
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </section>
            <section className="p-3 flex-1">
              <h2 className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Reglas para el lenguaje
              </h2>
              <p className="text-slate-400 text-base leading-relaxed">
                {levelConfig.rule}
              </p>
            </section>
          </aside>
        )}

        {/* Canvas ReactFlow */}
        <div ref={flowContainerRef} className="flex-1 relative min-w-0 bg-black">
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
          >
            <Background color="#334155" gap={16} size={0.5} />
          </ReactFlow>
        </div>
      </div>

      {/* Footer */}
      <footer className="shrink-0 bg-slate-800 border-t border-slate-700 px-4 py-3">
        <section>
          <p className="text-base font-semibold text-slate-400 uppercase tracking-wider mb-1">
            El resultado se mostrará acá
          </p>
          <p className="text-2xl font-semibold text-white min-h-[1.5rem]">
            {executedResult !== null ? executedResult : '—'}
          </p>
        </section>
      </footer>
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
    <NodeProvider flowContainerRef={flowContainerRef}>
      <DataflowContent
        isSandbox={isSandbox}
        levelConfig={levelConfig}
        backTo={backTo}
        flowContainerRef={flowContainerRef}
      />
    </NodeProvider>
  );
}
