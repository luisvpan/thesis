import { resolveStickColor, type ShapeColor, type ShapeSize } from '@/types/card-types';
import type { DataflowNode } from '@/contexts/node/types';
import { useNode } from '@/contexts/NodeContext';
import type { OperatorFlowNodeData } from './OperatorFlowNode';
import type { ProgramOutputFlowNodeData } from './ProgramOutputFlowNode';
import { ResultArrayVisual } from './ResultArrayVisual';
import type { SourceFlowNodeData } from './SourceFlowNode';
import { formatFractionText, type ResultViewMode } from './dataflowResultCpa';
import { CapGlyph, MontessoriCubeGlyph, StickGlyph } from './CpaGlyphs';
import { MiniShapeGlyph } from './MiniShapeGlyph';
import { foodEmoji } from '@/data/foodEmoji';
import { DiceFace } from './source-flow/DiceFace';
import { isPictorialColorYoloClass } from '@/data/pictorialColors';
import { getOrderedArrayZoneMembers } from '@/utils/arrayZoneGeometry';
import type { ResultValue } from '@/services/executeProgram';
import { numericValueOf } from '@/utils/resultValueDisplay';

const ARRAY_STRIP_SHELL =
  'flex max-w-[min(28rem,85vw)] items-center gap-1 px-1.5 py-1 shadow-lg pointer-events-none';
const ARRAY_BRACKET_CLASS = 'shrink-0 font-mono text-sm font-black text-teal-400';
const ARRAY_MEMBER_CLASS = 'flex h-9 min-w-9 shrink-0 items-center justify-center';

/** Serie numérica entre corchetes: el glyph de un arreglo en la conexión. */
function ArrayStrip({ children, empty }: { children: React.ReactNode; empty: boolean }) {
  return (
    <div className={ARRAY_STRIP_SHELL}>
      <span className={ARRAY_BRACKET_CLASS} aria-hidden>
        [
      </span>
      {empty ? <span className="px-1 text-xs italic text-slate-500">vacío</span> : children}
      <span className={ARRAY_BRACKET_CLASS} aria-hidden>
        ]
      </span>
    </div>
  );
}

function SourceMiniContent({
  data,
  viewMode,
}: {
  data: SourceFlowNodeData;
  viewMode: ResultViewMode;
}) {
  const pictorico = viewMode === 'pictorico';

  if (data.variant === 'number') {
    return (
      <span className="text-xl font-bold text-blue-400 tabular-nums">{data.value}</span>
    );
  }

  switch (data.variant) {
    case 'montessori':
      return <MontessoriCubeGlyph color={data.color ?? 'azul'} generic={false} />;
    case 'cap':
      return <CapGlyph color={data.color ?? 'azul'} generic={false} />;
    case 'stick':
      return (
        <StickGlyph
          color={resolveStickColor(data.color, data.yoloClass)}
          generic={false}
        />
      );
    case 'food':
      return (
        <span className="text-2xl leading-none" role="img" aria-label={data.food}>
          {foodEmoji(data.food ?? 'manzana')}
        </span>
      );
    case 'shape': {
      const shape = data.shape ?? 'circulo';
      const size: ShapeSize = data.size ?? 'mediano';
      let color: ShapeColor = data.color ?? 'amarillo';
      if (isPictorialColorYoloClass(data.yoloClass)) {
        color = data.color;
      }
      return (
        <MiniShapeGlyph shape={shape} size={size} color={color} generic={pictorico} />
      );
    }
    default:
      return null;
  }
}

function EvalResultMiniToken({ result }: { result: ResultValue }) {
  if (result.kind === 'semantic') {
    const { visualStrip, description } = result.result;
    if (visualStrip.length > 0) {
      return (
        <div className="max-w-48 scale-75 origin-center">
          <ResultArrayVisual items={visualStrip.slice(0, 6)} align="start" />
        </div>
      );
    }
    return (
      <span className="max-w-32 truncate text-xs font-medium text-slate-300">
        {description}
      </span>
    );
  }

  if (result.kind === 'numberArray') {
    return (
      <ArrayStrip empty={result.values.length === 0}>
        {result.values.map((item, i) => (
          <span key={i} className={`${ARRAY_MEMBER_CLASS} text-xl font-bold tabular-nums text-teal-300`}>
            {formatFractionText(item.numerator, item.denominator)}
          </span>
        ))}
      </ArrayStrip>
    );
  }

  const value = numericValueOf(result);
  if (value === undefined) return null;

  return <span className="text-xl font-bold text-teal-300 tabular-nums">{value}</span>;
}

/** El resultado de este nodo: el de la última ejecución, o el que guarda su carta. */
function resolveEvalResult(
  node: DataflowNode,
  evalResults: Map<string, ResultValue>
): ResultValue | null {
  const fromEval = evalResults.get(node.id);
  if (fromEval) return fromEval;

  if (node.type === 'operator' || node.type === 'programOutput') {
    const d = node.data as ProgramOutputFlowNodeData | OperatorFlowNodeData;
    return d.resultValue ?? null;
  }
  return null;
}

function NodeMiniVisual({
  node,
  viewMode,
  evalResults,
}: {
  node: DataflowNode;
  viewMode: ResultViewMode;
  evalResults: Map<string, ResultValue>;
}) {
  if (node.type === 'source') {
    return <SourceMiniContent data={node.data as SourceFlowNodeData} viewMode={viewMode} />;
  }
  const result = resolveEvalResult(node, evalResults);
  return result ? <EvalResultMiniToken result={result} /> : null;
}

function ArrayCloseMiniToken({
  closeNodeId,
  viewMode,
}: {
  closeNodeId: string;
  viewMode: ResultViewMode;
}) {
  const { nodes, edges, evalResults } = useNode();
  const members = getOrderedArrayZoneMembers(closeNodeId, nodes, edges);

  return (
    <ArrayStrip empty={members.length === 0}>
      {members.map((member) => (
        <div key={member.id} className={ARRAY_MEMBER_CLASS}>
          <NodeMiniVisual node={member} viewMode={viewMode} evalResults={evalResults} />
        </div>
      ))}
    </ArrayStrip>
  );
}

export function hasEdgeSourceMiniToken(
  node: DataflowNode,
  evalResults: Map<string, ResultValue>
): boolean {
  if (node.type === 'source' || node.type === 'arrayClose' || node.type === 'diceZone') return true;
  if (node.type === 'operator' || node.type === 'programOutput') {
    return resolveEvalResult(node, evalResults) !== null;
  }
  return false;
}

type EdgeSourceMiniTokenProps = {
  node: DataflowNode;
  viewMode: ResultViewMode;
};

export function EdgeSourceMiniToken({ node, viewMode }: EdgeSourceMiniTokenProps) {
  const { evalResults } = useNode();

  if (node.type === 'diceZone') {
    const data = node.data as { value?: number };
    return (
      <div className="scale-50 origin-center opacity-70">
        <DiceFace value={data.value} />
      </div>
    );
  }

  if (node.type === 'source') {
    const data = node.data as SourceFlowNodeData;
    return (
      <div>
        <SourceMiniContent data={data} viewMode={viewMode} />
      </div>
    );
  }

  if (node.type === 'arrayClose') {
    return <ArrayCloseMiniToken closeNodeId={node.id} viewMode={viewMode} />;
  }

  if (node.type === 'operator' || node.type === 'programOutput') {
    const result = resolveEvalResult(node, evalResults);
    if (!result) return null;
    return (
      <div className="flex max-w-[min(20rem,80vw)] items-center justify-center rounded-md px-2 py-1 shadow-lg">
        <EvalResultMiniToken result={result} />
      </div>
    );
  }

  return null;
}
