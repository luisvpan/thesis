import { useEffect, useState, type ReactNode } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { Hourglass, Loader2, Volume2 } from 'lucide-react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { ClickableHandle } from './ClickableHandle';
import {
  formatResultCpa,
  formatFraction,
  formatFractionText,
  type ResultViewMode,
} from './dataflowResultCpa';
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
import { speakSpanish, type SpeechStatus } from '@/utils/speakSpanish';
import { buildSinkResultSpeechText } from '@/utils/sinkResultSpeech';
import { describeCountedNoun } from '@/utils/spanishGrammar';
// Imports for result rendering heuristics - available for future use
import {
  computeMultiplicationGrouping,
  computeDivisionGrouping,
  type MultiplicationGrouping,
  type DivisionGrouping,
} from './result-rendering-heuristics';

/** Item in an ordered number array */
export type NumberArrayDisplayItem = {
  value: number;
  numerator: string;
  denominator: string;
};

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
  /** Ordered array of abstract numbers (e.g., from order) */
  numberArrayValues?: NumberArrayDisplayItem[];
  /** Resultado booleano (p. ej. compare). */
  booleanValue?: boolean;
};

export type ProgramOutputFlowNode = Node<ProgramOutputFlowNodeData, 'programOutput'>;

const MAX_GLYPHS = 36;

/** Estilo compartido para las respuestas en texto (descripción semántica y encabezado de objeto único). */
const ANSWER_TEXT_CLASS = 'text-2xl font-black leading-snug tracking-wide text-teal-100';

function SingleCpaGlyphStrip({
  meta,
  viewMode,
}: {
  meta: SingleCpaObjectMeta;
  viewMode: ResultViewMode;
}) {
  const { type, subtype, color, size, quantity } = meta;
  const count = Math.min(quantity, MAX_GLYPHS);
  const overflow = quantity - count;
  const generic = viewMode === 'pictorico';

  const glyphs = Array.from({ length: count }, (_, i) => {
    const key = `glyph-${i}`;
    switch (type) {
      case 'montessori':
        return <MontessoriCubeGlyph key={key} color={color} generic={generic} large />;
      case 'cap':
        return <CapGlyph key={key} color={color} generic={generic} large />;
      case 'stick':
        return <StickGlyph key={key} color={color} generic={generic} large />;
      case 'forma':
        return (
          <FormaGlyph
            key={key}
            subtype={subtype}
            color={color}
            size={size}
            generic={generic}
            large
          />
        );
      case 'comida':
        return <ComidaGlyph key={key} subtype={subtype} color={color} generic={generic} large />;
      default:
        return null;
    }
  });

  if (count === 0) {
    return <span className="text-slate-500 text-sm italic">vacío</span>;
  }

  return (
    <div className="flex flex-col items-start gap-1">
      <div className="flex flex-wrap justify-start gap-2.5">{glyphs}</div>
      {overflow > 0 ? (
        <span className="text-[10px] font-medium text-slate-400">+{overflow} más</span>
      ) : null}
    </div>
  );
}

// Para montessori/cap/stick el "subtipo" real de la frase es el tipo (cubo,
// tapa, paleta); el color de meta.color ya coincide con el usado internamente.
const NOUN_KEY_BY_TYPE: Record<string, string> = {
  montessori: 'montessori',
  cap: 'cap',
  stick: 'stick',
};

function singleCpaHeaderText(meta: SingleCpaObjectMeta, viewMode: ResultViewMode): string {
  if (viewMode === 'abstracto') {
    return String(meta.quantity);
  }
  const nounKey = NOUN_KEY_BY_TYPE[meta.type] ?? meta.subtype;
  const phrase = describeCountedNoun(nounKey, meta.quantity, {
    size: meta.size,
    color: meta.color,
  });
  return `${meta.quantity} ${phrase}`.trim();
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
        <span className="text-sm font-semibold text-red-400">Error</span>
      ),
    };
  }

  if (data.booleanValue !== undefined) {
    return {
      headerRight: (
        <span className="text-3xl font-black uppercase tracking-wide text-white">
          {data.booleanValue ? 'verdadero' : 'falso'}
        </span>
      ),
    };
  }

  // Ordered array of abstract numbers (e.g., from order)
  if (data.numberArrayValues && data.numberArrayValues.length > 0) {
    const formatted = data.numberArrayValues.map((item) =>
      formatFractionText(item.numerator, item.denominator)
    );
    return {
      headerRight: (
        <span className="text-2xl font-black tabular-nums text-white">
          [{formatted.join(', ')}]
        </span>
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
          <span className="text-6xl font-black tabular-nums text-white">
            {formatFraction(num, den)}
          </span>
        ),
      };
    }
    return {
      headerRight: (
        <span className={ANSWER_TEXT_CLASS}>{singleCpaHeaderText(meta, viewMode)}</span>
      ),
      resultVisual: <SingleCpaGlyphStrip meta={meta} viewMode={viewMode} />,
    };
  }

  if (data.description) {
    return {
      headerRight: <span className={ANSWER_TEXT_CLASS}>{data.description}</span>,
      resultVisual:
        data.visualStrip && data.visualStrip.length > 0 ? (
          <ResultArrayVisual items={data.visualStrip} align="start" large />
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
  const [speechStatus, setSpeechStatus] = useState<SpeechStatus>('idle');
  const isSpeechBusy = speechStatus === 'loading' || speechStatus === 'speaking';

  const speechButton = (
    <button
      type="button"
      disabled={!speechText || isSpeechBusy}
      onClick={() => {
        if (speechText) void speakSpanish(speechText, setSpeechStatus);
      }}
      className={`nodrag nopan ${FLOW_NODE_INTERACTIVE_CLASS} w-24 h-24 relative z-30 flex shrink-0 items-center justify-center gap-2 rounded-lg border-2 border-teal-600 bg-teal-800 px-2 py-2 text-sm font-semibold text-teal-50 shadow transition-colors hover:bg-teal-700 disabled:cursor-not-allowed disabled:opacity-60`}
      title={
        speechText
          ? speechStatus === 'loading'
            ? 'Preparando la voz…'
            : 'Escuchar el resultado'
          : 'Sin resultado para reproducir'
      }
    >
      {speechStatus === 'loading' ? (
        <Loader2
          className="h-12 w-12 shrink-0 animate-spin pointer-events-none"
          strokeWidth={2}
          aria-hidden
        />
      ) : (
        <Volume2
          className={`h-12 w-12 shrink-0 pointer-events-none ${speechStatus === 'speaking' ? 'animate-pulse' : ''}`}
          strokeWidth={2}
          aria-hidden
        />
      )}
    </button>
  );

  return (
    <div
      className={`relative flex w-70 -translate-x-[10%] -translate-y-[13%] ${shellClass}`}
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
        <SinkFlowNodeCard
          headerRight={headerRight}
          resultVisual={resultVisual}
          actionButton={speechButton}
        />
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
    </div>
  );
}
