import { useMemo } from 'react';
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

/**
 * Renders CPA glyphs organized into visual groups (for multiplication results).
 */
function GroupedCpaGlyphStrip({
  meta,
  viewMode,
  groupSize,
  groupCount,
}: {
  meta: SingleCpaObjectMeta;
  viewMode: ResultViewMode;
  groupSize: number;
  groupCount: number;
}) {
  const { type, subtype, color } = meta;
  const generic = viewMode === 'pictorico';
  const totalGlyphs = groupSize * groupCount;
  const maxTotal = MAX_GLYPHS;

  const renderGlyph = (key: string) => {
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
  };

  // Distribute glyphs across groups (respecting max)
  let rendered = 0;
  const groups: React.ReactNode[] = [];

  for (let g = 0; g < groupCount && rendered < maxTotal; g++) {
    const glyphsInGroup = Math.min(groupSize, maxTotal - rendered);
    groups.push(
      <div
        key={`group-${g}`}
        className="flex flex-wrap justify-center gap-1 p-1.5 rounded-md bg-slate-700/40 ring-1 ring-slate-600/50"
      >
        {Array.from({ length: glyphsInGroup }, (_, i) => renderGlyph(`g${g}-${i}`))}
      </div>
    );
    rendered += glyphsInGroup;
  }

  const overflow = totalGlyphs - rendered;

  if (groups.length === 0) {
    return <span className="text-slate-500 text-lg italic">vacío</span>;
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex flex-wrap justify-center gap-4 max-w-52">
        {groups}
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
  const { executionError, nodes, edges } = useNode();
  const { viewMode, orderingStrategy } = useResultCardUi();

  const value = data.value;
  const description = data.description;
  const visualStrip = data.visualStrip;

  const modeLabel = viewMode === 'pictorico' ? 'P' : viewMode === 'concreto' ? 'C' : 'A';

  // Detect if this output comes from an ordering operation
  const orderOperation = useMemo(() => {
    const inputEdge = edges.find((e) => e.target === id && e.targetHandle === 'in');
    if (!inputEdge) return null;

    const sourceNode = nodes.find((n) => n.id === inputEdge.source);
    if (!sourceNode || sourceNode.type !== 'operator') return null;

    const op = (sourceNode.data as { operation?: string }).operation;
    if (op === 'order_asc') return 'asc' as const;
    if (op === 'order_desc') return 'desc' as const;
    return null;
  }, [id, nodes, edges]);

  // Apply numerical post-ordering if strategy is 'numerical' and this is an ordering operation
  const { effectiveVisualStrip, effectiveDescription } = useMemo(() => {
    if (
      orderOperation === null ||
      orderingStrategy !== 'numerical' ||
      !data.originalElements?.length
    ) {
      return { effectiveVisualStrip: visualStrip, effectiveDescription: description };
    }

    const sorted = sortByQuantity(data.originalElements, orderOperation);
    return {
      effectiveVisualStrip: buildVisualStripFromElements(sorted),
      effectiveDescription: generateDescriptionFromElements(sorted),
    };
  }, [orderOperation, orderingStrategy, visualStrip, description, data.originalElements]);

  // Compute grouping for single CPA objects
  const grouping: MultiplicationGrouping = useMemo(() => {
    // Only applies in pictoric/concrete modes
    if (viewMode === 'abstracto') return { kind: 'none' };

    // For single CPA objects
    if (data.isSingleCpaObject && data.singleCpaObjectMeta) {
      const inputEdge = edges.find((e) => e.target === id && e.targetHandle === 'in');
      if (!inputEdge) return { kind: 'none' };

      const sourceNode = nodes.find((n) => n.id === inputEdge.source);
      if (!sourceNode) return { kind: 'none' };

      return computeMultiplicationGrouping(
        sourceNode,
        edges,
        nodes,
        data.singleCpaObjectMeta.quantity
      );
    }

    return { kind: 'none' };
  }, [viewMode, data, id, nodes, edges]);

  // Compute division grouping for single CPA objects
  const divisionGrouping: DivisionGrouping = useMemo(() => {
    if (viewMode === 'abstracto') return { kind: 'none' };

    if (data.isSingleCpaObject && data.singleCpaObjectMeta) {
      const inputEdge = edges.find((e) => e.target === id && e.targetHandle === 'in');
      if (!inputEdge) return { kind: 'none' };

      const sourceNode = nodes.find((n) => n.id === inputEdge.source);
      if (!sourceNode) return { kind: 'none' };

      return computeDivisionGrouping(
        sourceNode,
        edges,
        nodes,
        data.singleCpaObjectMeta.quantity
      );
    }

    return { kind: 'none' };
  }, [viewMode, data, id, nodes, edges]);

  // Compute grouping for array results
  const arrayGrouping = useMemo(() => {
    if (viewMode === 'abstracto' || !effectiveVisualStrip?.length) return undefined;

    const inputEdge = edges.find((e) => e.target === id && e.targetHandle === 'in');
    if (!inputEdge) return undefined;

    const sourceNode = nodes.find((n) => n.id === inputEdge.source);
    if (!sourceNode) return undefined;

    const result = computeMultiplicationGrouping(sourceNode, edges, nodes, effectiveVisualStrip.length);

    return result.kind === 'grouped'
      ? { groupSize: result.groupSize, groupCount: result.groupCount }
      : undefined;
  }, [viewMode, effectiveVisualStrip, id, nodes, edges]);

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
      ) : divisionGrouping.kind === 'grouped' ? (
        // Division grouped mode: show glyphs organized into groups (partitivo/cuotativo)
        <div className="flex flex-col items-center gap-1 text-white">
          <div className="flex items-center gap-1 text-slate-400">
            <Equal className="w-4 h-4" strokeWidth={2.5} />
            <span className="text-[10px] font-semibold uppercase tracking-wider">
              {modeLabel} · {divisionGrouping.mode === 'partitivo' ? 'Partitiva' : 'Cuotativa'}
            </span>
          </div>
          <GroupedCpaGlyphStrip
            meta={data.singleCpaObjectMeta}
            viewMode={viewMode}
            groupSize={divisionGrouping.groupSize}
            groupCount={divisionGrouping.groupCount}
          />
        </div>
      ) : grouping.kind === 'grouped' ? (
        // Multiplication grouped mode: show glyphs organized into visual groups
        <div className="flex flex-col items-center gap-1 text-white">
          <div className="flex items-center gap-1 text-slate-400">
            <Equal className="w-4 h-4" strokeWidth={2.5} />
            <span className="text-[10px] font-semibold uppercase tracking-wider">{modeLabel}</span>
          </div>
          <GroupedCpaGlyphStrip
            meta={data.singleCpaObjectMeta}
            viewMode={viewMode}
            groupSize={grouping.groupSize}
            groupCount={grouping.groupCount}
          />
        </div>
      ) : (
        // Non-grouped: show N glyphs without grouping
        <div className="flex flex-col items-center gap-1 text-white">
          <div className="flex items-center gap-1 text-slate-400">
            <Equal className="w-4 h-4" strokeWidth={2.5} />
            <span className="text-[10px] font-semibold uppercase tracking-wider">{modeLabel}</span>
          </div>
          <SingleCpaGlyphStrip meta={data.singleCpaObjectMeta} viewMode={viewMode} />
        </div>
      )
    ) : effectiveDescription ? (
      // Resultado semántico (arreglo de objetos)
      <div className="flex flex-col items-center gap-0.5 text-white">
        <LayoutList className="w-5 h-5 text-slate-400" strokeWidth={2} />
        <p className="text-lg text-center text-teal-200 leading-snug px-1">
          {effectiveDescription}
        </p>
        {effectiveVisualStrip && effectiveVisualStrip.length > 0 ? (
          <ResultArrayVisual items={effectiveVisualStrip} grouping={arrayGrouping} />
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
