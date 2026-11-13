import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ReactFlow,
  Background,
  type NodeTypes,
  type EdgeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  SourceFlowNode,
  OperatorFlowNode,
  ProgramOutputFlowNode,
  ArrayOpenNode,
  ArrayCloseNode,
  DiceZoneFlowNodeComponent,
  DataflowEdge,
  ArrayZoneEdge,
  type ResultViewMode,
} from '@/components/dataflow';
import { NodeProvider, useNode } from '@/contexts/NodeContext';
import { ResultCardUiProvider } from '@/contexts/ResultCardUiContext';
import { SocketStatusButton } from '@/components/SocketStatusButton';
import { ArrowLeft, Box, Eye, Shapes, Volume2 } from 'lucide-react';
import { MusicPlayerModal } from '@/components/MusicPlayerModal';

const nodeTypes: NodeTypes = {
  source: SourceFlowNode,
  operator: OperatorFlowNode,
  programOutput: ProgramOutputFlowNode,
  arrayOpen: ArrayOpenNode,
  arrayClose: ArrayCloseNode,
  diceZone: DiceZoneFlowNodeComponent,
};

const edgeTypes: EdgeTypes = {
  default: DataflowEdge,
  arrayZoneEdge: ArrayZoneEdge,
};

const VIEW_MODES: { id: ResultViewMode; label: string; icon: React.ReactNode }[] = [
  { id: 'concreto', label: 'Concreto', icon: <Box className="h-12 w-12 pointer-events-none" strokeWidth={2} aria-hidden /> },
  { id: 'pictorico', label: 'Pictórico', icon: <Shapes className="h-12 w-12 pointer-events-none" strokeWidth={2} aria-hidden /> },
  { id: 'abstracto', label: 'Abstracto', icon: <span className="text-4xl font-bold leading-none" aria-hidden>123</span> },
];

const PILL = 'flex items-center gap-2 rounded-[2.25rem] border border-slate-700/80 bg-slate-900/90 p-3 shadow-lg shadow-black/40';
const ICON_BTN = 'flex h-24 w-24 items-center justify-center rounded-3xl border-2 transition-colors';

export function DataflowContent({ flowContainerRef }: {
  flowContainerRef: React.RefObject<HTMLDivElement | null>;
}) {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    nodesDraggable,
    clearSelection,
  } = useNode();

  const [viewMode, setViewMode] = useState<ResultViewMode>('abstracto');

  useEffect(() => {
    clearSelection();
  }, [viewMode, clearSelection]);
  const [showOperatorResults, setShowOperatorResults] = useState(false);
  const [showMusicModal, setShowMusicModal] = useState(false);

  return (
    <div className="h-screen w-screen flex flex-col bg-black font-display">
      <header className="flex shrink-0 items-center justify-between gap-4 bg-black px-6 pt-6 pb-4">
        <div className={`${PILL} pr-6`}>
          <Link
            to="/"
            className={`${ICON_BTN} border-slate-600/70 bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white`}
            title="Volver al inicio"
          >
            <ArrowLeft className="h-12 w-12 pointer-events-none" strokeWidth={2.5} />
          </Link>
          <div className="min-w-0 pl-2">
            <h1 className="truncate text-4xl font-bold leading-tight text-white">Modo libre</h1>
         </div>
        </div>

        <div className={PILL} role="group" aria-label="Modo de visualización">
          {VIEW_MODES.map((mode) => (
            <button
              key={mode.id}
              type="button"
              onClick={() => setViewMode(mode.id)}
              aria-pressed={viewMode === mode.id}
              className={`flex h-24 items-center gap-4 rounded-3xl border-2 px-10 text-4xl font-semibold transition-colors ${
                viewMode === mode.id ? 'border-teal-400/70 bg-teal-700 text-white' : 'border-slate-600/70 text-slate-200 hover:bg-slate-800'
              }`}
            >
              {mode.icon}
              {mode.label}
            </button>
          ))}
        </div>

        <div className={PILL}>
          <button
            type="button"
            onClick={() => setShowOperatorResults((v) => !v)}
            aria-pressed={showOperatorResults}
            className={`${ICON_BTN} ${
              showOperatorResults ? 'border-teal-400/70 bg-teal-700 text-white' : 'border-slate-600/70 bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white'
            }`}
            title={showOperatorResults ? 'Ocultar resultados' : 'Mostrar resultados'}
          >
            <Eye className="h-12 w-12 pointer-events-none" strokeWidth={2} aria-hidden />
          </button>
          <button
            type="button"
            onClick={() => setShowMusicModal(true)}
            className={`${ICON_BTN} border-slate-600/70 bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white`}
            title="Música"
          >
            <Volume2 className="h-12 w-12 pointer-events-none" strokeWidth={2} />
          </button>
          <SocketStatusButton className={ICON_BTN} />
        </div>
      </header>

      <div className="flex-1 flex min-h-0">
        {/* Canvas ReactFlow */}
        <div ref={flowContainerRef} className="flex-1 relative min-w-0 bg-black">
          <ResultCardUiProvider
            viewMode={viewMode}
            hasExecuted={true}
            showFlowResults={showOperatorResults}
          >
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
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

      {showMusicModal ? <MusicPlayerModal onClose={() => setShowMusicModal(false)} /> : null}
    </div>
  );
}

// Main component that provides NodeContext
export default function DataflowPage() {
  const flowContainerRef = useRef<HTMLDivElement>(null);

  return (
    <NodeProvider flowContainerRef={flowContainerRef}>
      <DataflowContent flowContainerRef={flowContainerRef} />
    </NodeProvider>
  );
}
