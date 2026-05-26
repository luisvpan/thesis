import { useEffect, type ReactNode } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { Hourglass, Volume2 } from 'lucide-react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa, formatFraction, type ResultViewMode } from './dataflowResultCpa';
import { SinkFlowNodeCard } from './SinkFlowNodeCard';
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
import { FLOW_NODE_INTERACTIVE_CLASS } from './flowNodeChrome';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';
import { speakSpanish } from '@/utils/speakSpanish';
import { buildSinkResultSpeechText } from '@/utils/sinkResultSpeech';
// Imports for result rendering heuristics - available for future use
import {
  computeMultiplicationGrouping,
  computeDivisionGrouping,
  type MultiplicationGrouping,
  type DivisionGrouping,
} from './result-rendering-heuristics';
import {
  sortByQuantity,
  buildVisualStripFromElements,
  generateDescriptionFromElements,
} from '@/utils/post-ordering';

/** Solo frontend: muestra salida tras ejecutar; valor numérico o descripción semántica. */
export type ProgramOutputFlowNodeData = VisionNodeMeta & {
  /** Valor numérico para resultados racionales */
  value?: number;
  /** Descripción semántica para resultados de arreglo */
  description?: string;
  /** Cubos / iconos en orden del arreglo (Montessori, forma, comida). */
  visualStrip?: ResultVisualItem[];
  /** Elementos originales sin expandir, para re-ordenamiento en frontend. */
  originalElements?: unknown[];
  /** Set to true when result is a single CPAObject (not an array) */
  isSingleCpaObject?: boolean;
  /** Metadata for single CPA object rendering */
  singleCpaObjectMeta?: SingleCpaObjectMeta;
  /** For exact fraction display of pure rationals (e.g., "13/4" instead of 3.25) */
  numerator?: string;
  denominator?: string;
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
        return (
          <FormaGlyph key={key} subtype={subtype} color={color} generic={generic} />
        );
      case 'comida':
        return <ComidaGlyph key={key} subtype={subtype} color={color} generic={generic} />;
      default:
        return null;
    }
  });

  if (count === 0) {
    return <span className="text-slate-500 text-sm italic">vacío</span>;
  }

  return (
    <div className="flex flex-col items-start gap-1">
      <div className="flex flex-wrap justify-start gap-1.5">{glyphs}</div>
      {overflow > 0 ? (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      ) : null}
    </div>
  );
}

function singleCpaHeaderText(meta: SingleCpaObjectMeta, viewMode: ResultViewMode): string {
  if (viewMode === 'abstracto') {
    return String(meta.quantity);
  }
  const colorPart = meta.color ? meta.color.toUpperCase() : '';
  const typeLabels: Record<string, string> = {
    montessori: 'cubos',
    cap: 'tapas',
    stick: 'palitos',
    forma: meta.subtype,
    comida: meta.subtype,
  };
  const label = typeLabels[meta.type] ?? meta.type;
  if (colorPart) {
    return `${meta.quantity} ${label} ${colorPart}`.trim();
  }
  return `${meta.quantity} ${label}`.trim();
}

type SinkBodyParts = {
  headerRight: ReactNode;
  resultVisual?: ReactNode;
};

function buildSinkBody(
  data: ProgramOutputFlowNodeData,
  executionError: string | null | undefined,
  viewMode: ResultViewMode
): SinkBodyParts {
  if (executionError) {
    return {
      headerRight: (
        <p className="text-sm font-semibold leading-snug text-red-400 whitespace-pre-wrap">
          {executionError}
        </p>
      ),
    };
  }

  if (data.isSingleCpaObject && data.singleCpaObjectMeta) {
    const meta = data.singleCpaObjectMeta;
    if (viewMode === 'abstracto') {
      const num = data.numerator ?? meta.numerator ?? String(meta.quantity);
      const den = data.denominator ?? meta.denominator ?? '1';
      return {
        headerRight: (
          <span className="text-3xl font-black tabular-nums text-white">
            {formatFraction(num, den)}
          </span>
        ),
      };
    }
    return {
      headerRight: singleCpaHeaderText(meta, viewMode),
      resultVisual: <SingleCpaGlyphStrip meta={meta} viewMode={viewMode} />,
    };
  }

  if (data.description) {
    return {
      headerRight: data.description,
      resultVisual:
        data.visualStrip && data.visualStrip.length > 0 ? (
          <ResultArrayVisual items={data.visualStrip} align="start" />
        ) : undefined,
    };
  }

  if (data.value !== undefined) {
    if (viewMode === 'pictorico' && Number.isInteger(data.value) && data.value >= 0 && data.value <= 24) {
      return {
        headerRight: <span className="tabular-nums text-white">{data.value}</span>,
        resultVisual: (
          <div className="flex justify-start">
            {formatResultCpa(data.value, viewMode, data.numerator, data.denominator)}
          </div>
        ),
      };
    }
    return {
      headerRight: (
        <span
          className={
            viewMode === 'abstracto'
              ? 'text-6xl font-black tabular-nums text-white'
              : 'text-lg font-bold text-sky-300'
          }
        >
          {formatResultCpa(data.value, viewMode, data.numerator, data.denominator)}
        </span>
      ),
    };
  }

  return {
    headerRight: (
      <span className="flex items-center justify-end gap-1.5 text-slate-500 italic">
        <Hourglass className="h-4 w-4 shrink-0" strokeWidth={1.5} />
        Sin resultado
      </span>
    ),
  };
}

// Re-export types for external use
export type { MultiplicationGrouping, DivisionGrouping };

// Re-export functions for external use
export {
  computeMultiplicationGrouping,
  computeDivisionGrouping,
  sortByQuantity,
  buildVisualStripFromElements,
  generateDescriptionFromElements,
};

export function ProgramOutputFlowNode({
  id,
  data,
}: NodeProps<ProgramOutputFlowNode>) {
  const { executionError, registerPortKind, unregisterPortKinds } = useNode();

  useEffect(() => {
    registerPortKind(id, 'in', { accepts: ['any'] });
    registerPortKind(id, 'out', { produces: 'any' });
    return () => unregisterPortKinds(id);
  }, [id, registerPortKind, unregisterPortKinds]);
  const { viewMode } = useResultCardUi();
  const shellClass = useFlowNodeShellClass();
  const { headerRight, resultVisual } = buildSinkBody(data, executionError, viewMode);
  const trackId = readTrackId(data);
  const speechText = buildSinkResultSpeechText(data, executionError, viewMode);

  return (
    <div
      className={`relative flex w-70 -translate-x-[10%] -translate-y-[40%] flex-col-reverse gap-2 ${shellClass}`}
    >
      <div className="pointer-events-none relative h-65 w-full">
        <TrackIdBadge trackId={trackId} />
        <ClickableHandle
          type="target"
          position={Position.Left}
          id="in"
          nodeId={id}
          handleVariant="sink-in"
          accepts={['any']}
          style={{ transform: 'translateX(-100px) translateY(-150%)' }}
        />
        <SinkFlowNodeCard headerRight={headerRight} resultVisual={resultVisual} />
        <ClickableHandle
          type="source"
          position={Position.Right}
          id="out"
          nodeId={id}
          handleVariant="sink-out"
          produces="any"
          style={{ transform: 'translateX(100px) translateY(-150%)' }}
        />
      </div>
      <button
        type="button"
        onClick={() => {
          if (speechText) speakSpanish(speechText);
        }}
        className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} w-20 h-20 relative z-30 flex shrink-0 items-center justify-center gap-2 rounded-lg border-2 border-teal-600 bg-teal-800 px-3 py-2 text-sm font-semibold text-teal-50 shadow transition-colors hover:bg-teal-700`}
        title={speechText ? 'Escuchar el resultado' : 'Sin resultado para reproducir'}
      >
        <Volume2 className="h-10 w-10 shrink-0" strokeWidth={2} aria-hidden />
      </button>
    </div>
  );
}
