import type { OperatorType, ShapeColor, ShapeSize } from '@/types/card-types';
import type { DataflowNode } from '@/contexts/node/types';
import { useNode } from '@/contexts/NodeContext';
import type { OperatorFlowNodeData } from './OperatorFlowNode';
import type { SourceFlowNodeData } from './SourceFlowNode';
import type { ResultViewMode } from './dataflowResultCpa';
import { CapGlyph, MontessoriCubeGlyph, StickGlyph } from './CpaGlyphs';
import { MiniShapeGlyph } from './MiniShapeGlyph';
import { foodEmoji } from '@/data/foodEmoji';
import { isPictorialColorYoloClass } from '@/data/pictorialColors';
import { getOrderedArrayZoneMembers } from '@/utils/arrayZoneGeometry';

const TOKEN_SHELL =
  'flex h-11 w-11 items-center justify-center rounded-lg border-2 shadow-lg bg-slate-900/90 pointer-events-none';

const ARRAY_STRIP_SHELL =
  'flex max-w-[min(28rem,85vw)] items-center gap-1 px-1.5 py-1 shadow-lg pointer-events-none';

function operatorSymbol(operator: OperatorType): string {
  if (operator === 'adicion') return '+';
  if (operator === 'sustraccion') return '−';
  if (operator === 'multiplicacion') return '×';
  if (operator === 'division') return '÷';
  if (operator === 'orden-menor-mayor') return '↑';
  if (operator === 'orden-mayor-menor') return '↓';
  if (operator === 'filtrar-general') return '⊲';
  return '?';
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
      return <StickGlyph color={data.color ?? 'rojo'} generic={false} />;
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

function NodeMiniVisual({ node, viewMode }: { node: DataflowNode; viewMode: ResultViewMode }) {
  if (node.type === 'source') {
    return <SourceMiniContent data={node.data as SourceFlowNodeData} viewMode={viewMode} />;
  }
  if (node.type === 'operator') {
    const operator = (node.data as OperatorFlowNodeData).operator ?? 'adicion';
    return (
      <span className="text-xl font-bold text-red-400">{operatorSymbol(operator)}</span>
    );
  }
  return null;
}

function ArrayCloseMiniToken({
  closeNodeId,
  viewMode,
}: {
  closeNodeId: string;
  viewMode: ResultViewMode;
}) {
  const { nodes, edges } = useNode();
  const members = getOrderedArrayZoneMembers(closeNodeId, nodes, edges);

  return (
    <div className={ARRAY_STRIP_SHELL}>
      <span className="shrink-0 font-mono text-sm font-black text-teal-400" aria-hidden>
        [
      </span>
      {members.length === 0 ? (
        <span className="px-1 text-xs italic text-slate-500">vacío</span>
      ) : (
        members.map((member) => (
          <div
            key={member.id}
            className="flex h-9 w-9 shrink-0 items-center justify-center"
          >
            <NodeMiniVisual node={member} viewMode={viewMode} />
          </div>
        ))
      )}
      <span className="shrink-0 font-mono text-sm font-black text-teal-400" aria-hidden>
        ]
      </span>
    </div>
  );
}

type EdgeSourceMiniTokenProps = {
  node: DataflowNode;
  viewMode: ResultViewMode;
};

export function EdgeSourceMiniToken({ node, viewMode }: EdgeSourceMiniTokenProps) {
  if (node.type === 'source') {
    const data = node.data as SourceFlowNodeData;

    return (
      <div className={``}>
        <SourceMiniContent data={data} viewMode={viewMode} />
      </div>
    );
  }

  if (node.type === 'arrayClose') {
    return <ArrayCloseMiniToken closeNodeId={node.id} viewMode={viewMode} />;
  }

  return null;
}
