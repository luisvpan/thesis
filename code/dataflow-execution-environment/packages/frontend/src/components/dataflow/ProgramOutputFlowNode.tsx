import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { Equal, LayoutList, Hourglass } from 'lucide-react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa, type ResultViewMode } from './dataflowResultCpa';
import { FlowNodeCard } from './FlowNodeCard';
import { ResultArrayVisual } from './ResultArrayVisual';
import {
  MontessoriCubeGlyph,
  CapGlyph,
  StickGlyph,
  FormaGlyph,
  ComidaGlyph,
} from './CpaGlyphs';
import type { ResultVisualItem, SingleCpaObjectMeta } from '@/services/executeProgram';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';

/** Solo frontend: muestra salida tras ejecutar; valor numérico o descripción semántica. */
export type ProgramOutputFlowNodeData = VisionNodeMeta & {
  /** Valor numérico para resultados racionales */
  value?: number;
  /** Descripción semántica para resultados de arreglo */
  description?: string;
  /** Cubos / iconos en orden del arreglo (Montessori, forma, comida). */
  visualStrip?: ResultVisualItem[];
  /** Set to true when result is a single CPAObject (not an array) */
  isSingleCpaObject?: boolean;
  /** Metadata for single CPA object rendering */
  singleCpaObjectMeta?: SingleCpaObjectMeta;
};

export type ProgramOutputFlowNode = Node<ProgramOutputFlowNodeData, 'programOutput'>;

const MAX_GLYPHS = 36;

function SingleCpaGlyphStrip({
  meta,
  viewMode,
}: {
  meta: SingleCpaObjectMeta;
  viewMode: ResultViewMode;
}) {
  const { type, subtype, color, quantity } = meta;
  const count = Math.min(quantity, MAX_GLYPHS);
  const overflow = quantity - count;

  // In "pictorico" mode, use generic (teal) appearance
  const generic = viewMode === 'pictorico';

  const glyphs = Array.from({ length: count }, (_, i) => {
    const key = `glyph-${i}`;
    switch (type) {
      case 'montessori':
        return <MontessoriCubeGlyph key={key} color={color} generic={generic} />;
      case 'cap':
        return <CapGlyph key={key} color={color} generic={generic} />;
      case 'stick':
        return <StickGlyph key={key} color={color} generic={generic} />;
      case 'forma':
        return <FormaGlyph key={key} subtype={subtype} generic={generic} />;
      case 'comida':
        return <ComidaGlyph key={key} subtype={subtype} color={color} generic={generic} />;
      default:
        return null;
    }
  });

  if (count === 0) {
    return <span className="text-slate-500 text-lg italic">vacío</span>;
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex flex-wrap justify-center gap-1.5 max-w-44">
        {glyphs}
      </div>
      {overflow > 0 && (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      )}
    </div>
  );
}

export function ProgramOutputFlowNode({
  id,
  data,
}: NodeProps<ProgramOutputFlowNode>) {
  const { executionError } = useNode();
  const { viewMode } = useResultCardUi();

  const value = data.value;
  const description = data.description;
  const visualStrip = data.visualStrip;

  const modeLabel = viewMode === 'pictorico' ? 'P' : viewMode === 'concreto' ? 'C' : 'A';

  const display =
    executionError ? (
      <p className="max-h-48 overflow-y-auto text-left text-sm font-semibold leading-snug text-red-400 whitespace-pre-wrap px-1">
        {executionError}
      </p>
    ) : (data.isSingleCpaObject && data.singleCpaObjectMeta) ? (
      // Single CPA Object - viewMode-aware rendering
      viewMode === 'abstracto' ? (
        // Abstract mode: just show the quantity as a number
        <div className="flex flex-col items-center gap-1 text-white">
          <div className="flex items-center gap-1 text-slate-400">
            <Equal className="w-4 h-4" strokeWidth={2.5} />
            <span className="text-[10px] font-semibold uppercase tracking-wider">{modeLabel}</span>
          </div>
          <div className="text-5xl font-black text-white tabular-nums drop-shadow-lg">
            {data.singleCpaObjectMeta.quantity}
          </div>
        </div>
      ) : (
        // Pictoric or Concrete mode: show N glyphs
        <div className="flex flex-col items-center gap-1 text-white">
          <div className="flex items-center gap-1 text-slate-400">
            <Equal className="w-4 h-4" strokeWidth={2.5} />
            <span className="text-[10px] font-semibold uppercase tracking-wider">{modeLabel}</span>
          </div>
          <SingleCpaGlyphStrip meta={data.singleCpaObjectMeta} viewMode={viewMode} />
        </div>
      )
    ) : description ? (
      // Resultado semántico (arreglo de objetos)
      <div className="flex flex-col items-center gap-0.5 text-white">
        <LayoutList className="w-5 h-5 text-slate-400" strokeWidth={2} />
        <p className="text-lg text-center text-teal-200 leading-snug px-1">
          {description}
        </p>
        {visualStrip && visualStrip.length > 0 ? (
          <ResultArrayVisual items={visualStrip} />
        ) : null}
      </div>
    ) : value !== undefined ? (
      // Resultado numérico
      <div className="flex flex-col items-center gap-1 text-white">
        <div className="flex items-center gap-1 text-slate-400">
          <Equal className="w-4 h-4" strokeWidth={2.5} />
          <span className="text-[10px] font-semibold uppercase tracking-wider">{modeLabel}</span>
        </div>
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
      <div className="flex flex-col items-center gap-2 text-slate-500">
        <Hourglass className="w-6 h-6" strokeWidth={1.5} />
        <p className="text-sm text-center italic px-2">Sin resultado</p>
      </div>
    );

  const trackId = readTrackId(data);

  return (
    <div className="relative h-65 w-52 -translate-x-[30%] -translate-y-[80%]">
      <TrackIdBadge trackId={trackId} />
      <ClickableHandle type="target" position={Position.Left} id="in" nodeId={id} style={{ transform: 'translateX(-100px)' }} />
      <FlowNodeCard family="sink" title="Salida" content={<span className="text-xs font-black text-slate-100">{display}</span>} />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} style={{ transform: 'translateX(100px)' }} />
    </div>
  );
}
