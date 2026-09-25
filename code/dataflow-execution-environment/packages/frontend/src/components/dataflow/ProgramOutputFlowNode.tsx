import { Fragment, useEffect, useState, type ReactNode } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { Hourglass, Loader2, TriangleAlert, Volume2 } from 'lucide-react';
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
import { FractionGlyph } from './FractionGlyph';
import { isDrawableFraction } from './fractionGeometry';
import type { OutputErrorInfo, SingleCpaObjectMeta } from '@/services/executeProgram';
import type { WithResultValue } from '@/utils/resultValueDisplay';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import type { NodeErrorMark } from '@/contexts/node/errorMarks';
import { FLOW_NODE_INTERACTIVE_CLASS } from './flowNodeChrome';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';
import { speakSpanish, type SpeechStatus } from '@/utils/speakSpanish';
import { buildSinkResultSpeechText } from '@/utils/sinkResultSpeech';
import { describeCountedNoun } from '@/utils/spanishGrammar';

/** Solo frontend: lo que muestra una salida tras ejecutar. */
export type ProgramOutputFlowNodeData = VisionNodeMeta &
  WithResultValue & {
    /** Los errores que apagaron esta salida en la última ejecución (§4). */
    errors?: OutputErrorInfo[];
    /** Papel de la carta en ese error; en una salida, siempre `sink`. */
    errorMark?: NodeErrorMark;
  };

export type ProgramOutputFlowNode = Node<ProgramOutputFlowNodeData, 'programOutput'>;

const MAX_GLYPHS = 36;

/** Estilo compartido para las respuestas en texto (descripción semántica y encabezado de objeto único). */
const ANSWER_TEXT_CLASS = 'text-2xl font-black leading-snug tracking-wide text-teal-100';
/** Mismo tamaño que ANSWER_TEXT_CLASS, en tono de error. */
const ERROR_TEXT_CLASS = 'text-2xl font-black leading-snug tracking-wide text-red-400';

/** Lo que sobra del último objeto, como `n` de `d` regiones; `null` si está entero. */
function partialOf(meta: SingleCpaObjectMeta): { numerator: number; denominator: number } | null {
  const denominator = Number(meta.denominator);
  const numerator = Number(meta.numerator) % denominator;
  return isDrawableFraction(numerator, denominator) ? { numerator, denominator } : null;
}

function SingleCpaGlyphStrip({
  meta,
  viewMode,
}: {
  meta: SingleCpaObjectMeta;
  viewMode: ResultViewMode;
}) {
  const { type, subtype, color, size, quantity } = meta;
  const whole = Math.min(Math.max(0, Math.floor(quantity)), MAX_GLYPHS);
  const overflow = Math.max(0, Math.floor(quantity)) - whole;
  const generic = viewMode === 'pictorico';

  const glyph = () => {
    switch (type) {
      case 'montessori':
        return <MontessoriCubeGlyph color={color} generic={generic} large />;
      case 'cap':
        return <CapGlyph color={color} generic={generic} large />;
      case 'stick':
        return <StickGlyph color={color} generic={generic} large />;
      case 'forma':
        return <FormaGlyph subtype={subtype} color={color} size={size} generic={generic} large />;
      case 'comida':
        return <ComidaGlyph subtype={subtype} color={color} generic={generic} large />;
      default:
        return null;
    }
  };

  const glyphs: ReactNode[] = Array.from({ length: whole }, (_, i) => (
    <Fragment key={`glyph-${i}`}>{glyph()}</Fragment>
  ));

  // El último va incompleto cuando la cantidad no es entera: 3/2 manzanas son
  // una manzana y otra a la que le falta la mitad.
  const partial = partialOf(meta);
  if (partial) {
    glyphs.push(
      <FractionGlyph key="glyph-parcial" {...partial}>
        {glyph()}
      </FractionGlyph>
    );
  }

  if (glyphs.length === 0) {
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

/** Los errores que apagaron *esta* salida; los de otra no son asunto suyo (§4). */
function ownErrorText(data: ProgramOutputFlowNodeData): string | null {
  const errors = data.errors ?? [];
  if (errors.length === 0) return null;
  return errors.map((error) => error.text).join("\n");
}

function buildSinkBody(
  data: ProgramOutputFlowNodeData,
  programError: string | null | undefined,
  viewMode: ResultViewMode
): SinkBodyParts {
  const errorText = ownErrorText(data) ?? programError;

  if (errorText) {
    return {
      headerRight: (
        <div className="flex items-start gap-2">
          <TriangleAlert
            className="h-6 w-6 shrink-0 text-red-400"
            strokeWidth={2.5}
            aria-hidden
          />
          <span className={`${ERROR_TEXT_CLASS} whitespace-pre-line`}>{errorText}</span>
        </div>
      ),
    };
  }

  const result = data.resultValue;

  if (!result) {
    return {
      headerRight: (
        <span className="flex items-center justify-end gap-1.5 text-slate-500 italic">
          <Hourglass className="h-4 w-4 shrink-0" strokeWidth={1.5} />
          Sin resultado
        </span>
      ),
    };
  }

  // Una forma a la vez, y el compilador comprueba que estén todas: el orden de
  // las ramas ya no decide nada.
  switch (result.kind) {
    case 'boolean':
      return {
        headerRight: (
          <span className="text-3xl font-black uppercase tracking-wide text-white">
            {result.value ? 'verdadero' : 'falso'}
          </span>
        ),
      };

    case 'numberArray': {
      const formatted = result.values.map((item) =>
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

    case 'number': {
      const { value, numerator, denominator } = result;
      if (viewMode === 'pictorico' && Number.isInteger(value) && value >= 0 && value <= 24) {
        return {
          headerRight: <span className="tabular-nums text-white">{value}</span>,
          resultVisual: (
            <div className="flex justify-start">
              {formatResultCpa(value, viewMode, numerator, denominator)}
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
            {formatResultCpa(value, viewMode, numerator, denominator)}
          </span>
        ),
      };
    }

    case 'semantic': {
      // Un solo objeto se pinta como carta; varios, como grupo. La fracción sale
      // de su propio meta, que es donde vive.
      const meta = result.singleCpaObjectMeta;

      if (meta) {
        if (viewMode === 'abstracto') {
          return {
            headerRight: (
              <span className="text-6xl font-black tabular-nums text-white">
                {formatFraction(meta.numerator, meta.denominator)}
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

      const { description, visualStrip } = result.result;
      return {
        headerRight: <span className={ANSWER_TEXT_CLASS}>{description}</span>,
        resultVisual:
          visualStrip.length > 0 ? (
            <ResultArrayVisual items={visualStrip} align="start" large />
          ) : undefined,
      };
    }
  }
}

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
          errorMark={data.errorMark}
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
