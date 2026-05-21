import type { ReactNode } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { Hourglass } from 'lucide-react';
import { useNode } from '@/contexts/NodeContext';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { ClickableHandle } from './ClickableHandle';
import { formatResultCpa, type ResultViewMode } from './dataflowResultCpa';
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
      return {
        headerRight: (
          <span className="text-3xl font-black tabular-nums text-white">{meta.quantity}</span>
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
          <div className="flex justify-start">{formatResultCpa(data.value, viewMode)}</div>
        ),
      };
    }
    return {
      headerRight: (
        <span
          className={
            viewMode === 'abstracto'
              ? 'text-3xl font-black tabular-nums text-white'
              : 'text-lg font-bold text-sky-300'
          }
        >
          {formatResultCpa(data.value, viewMode)}
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

export function ProgramOutputFlowNode({
  id,
  data,
}: NodeProps<ProgramOutputFlowNode>) {
  const { executionError } = useNode();
  const { viewMode } = useResultCardUi();
  const { headerRight, resultVisual } = buildSinkBody(data, executionError, viewMode);
  const trackId = readTrackId(data);

  return (
    <div className="relative h-65 w-70 -translate-x-[10%] ">
      <TrackIdBadge trackId={trackId} />
      <ClickableHandle
        type="target"
        position={Position.Left}
        id="in"
        nodeId={id}
        style={{ transform: 'translateX(-100px) translateY(-200%)' }}
      />
      <SinkFlowNodeCard headerRight={headerRight} resultVisual={resultVisual} />
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
        style={{ transform: 'translateX(100px) translateY(-200%)' }}
      />
    </div>
  );
}
