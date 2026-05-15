import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { ShapeType, ShapeSize, ShapeColor, FoodType, MontessoriColor } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';
import { useResultCardUi } from '@/contexts/ResultCardUiContext';
import { isSourceBlockedByCpaMode } from '@/utils/cpaModeUtils';
import { CpaModeDisabledBanner } from './CpaModeDisabledBanner';

export type SourceFlowNodeData =
  | { variant: 'number'; value: number; visionSubtitle?: string; trackId?: number }
  | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor; trackId?: number }
  | { variant: 'food'; yoloClass: string; food: FoodType; trackId?: number }
  | { variant: 'montessori'; yoloClass: string; color: MontessoriColor; trackId?: number };

export type SourceFlowNode = Node<SourceFlowNodeData, 'source'>;

/** Cartas pictóricas de tamaño sin forma explícita (distintas de sm_/md_/lg_* + figura). */
const PICTORIAL_SIZE_ONLY_YOLO = new Set(['small', 'medium', 'large']);

function sourceTitle(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return 'Numero';
  if (data.variant === 'shape' && PICTORIAL_SIZE_ONLY_YOLO.has(data.yoloClass)) return 'Tamaño';
  if (data.variant === 'shape') return 'Forma';
  if (data.variant === 'montessori') return 'Montessori';
  return 'Comida';
}

function sourceMain(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return String(data.value);
  if (data.variant === 'shape' && PICTORIAL_SIZE_ONLY_YOLO.has(data.yoloClass)) return data.size;
  if (data.variant === 'shape') return `${data.shape} ${data.size}`;
  if (data.variant === 'montessori') return data.color;
  return data.food;
}

export function SourceFlowNode({ id, data }: NodeProps<SourceFlowNode>) {
  const d = (data ?? { variant: 'number', value: 0 }) as SourceFlowNodeData;
  const subtitle = d.variant === 'number' ? d.visionSubtitle : undefined;
  const { viewMode } = useResultCardUi();
  const blocked = isSourceBlockedByCpaMode(d, viewMode);

  return (
    <div className="relative h-80 w-52 -translate-x-[30%] -translate-y-[50%]">
      <FlowNodeCard
        family="input"
        title={sourceTitle(d)}
        content={<span className="text-xs font-black text-slate-100">{sourceMain(d)}</span>}
        subtitle={subtitle}
        topNotice={<CpaModeDisabledBanner show={blocked} />}
      />
      <ClickableHandle type="source" position={Position.Right} id="out" nodeId={id} style={{ transform: 'translateX(100px)' }} disabled={blocked} />
    </div>
  );
}
