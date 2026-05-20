import { useEffect } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { ShapeType, ShapeSize, ShapeColor, FoodType, MontessoriColor, CapColor, StickColor } from '@/types/card-types';
import { FlowNodeCard } from './FlowNodeCard';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { useNode } from '@/contexts/NodeContext';
import type { HandleKind } from './handle-kinds';

type VisionSynced = VisionNodeMeta;

export type SourceFlowNodeData = VisionSynced &
  (
    | { variant: 'number'; value: number; visionSubtitle?: string }
    | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
    | { variant: 'food'; yoloClass: string; food: FoodType }
    | { variant: 'montessori'; yoloClass: string; color: MontessoriColor }
    | { variant: 'cap'; yoloClass: string; color: CapColor }
    | { variant: 'stick'; yoloClass: string; color: StickColor }
  );

export type SourceFlowNode = Node<SourceFlowNodeData, 'source'>;

/** Cartas pictóricas de tamaño sin forma explícita (distintas de sm_/md_/lg_* + figura). */
const PICTORIAL_SIZE_ONLY_YOLO = new Set(['small', 'medium', 'large']);

function sourceTitle(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return 'Numero';
  if (data.variant === 'shape' && PICTORIAL_SIZE_ONLY_YOLO.has(data.yoloClass)) return 'Tamaño';
  if (data.variant === 'shape') return 'Forma';
  if (data.variant === 'montessori') return 'Cubo';
  if (data.variant === 'cap') return 'Tapa';
  if (data.variant === 'stick') return 'Palito';
  return 'Comida';
}

function sourceMain(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return String(data.value);
  if (data.variant === 'shape' && PICTORIAL_SIZE_ONLY_YOLO.has(data.yoloClass)) return data.size;
  if (data.variant === 'shape') return `${data.shape} ${data.size}`;
  if (data.variant === 'montessori') return data.color;
  if (data.variant === 'cap') return data.color;
  if (data.variant === 'stick') return data.color;
  return data.food;
}

export function SourceFlowNode({ id, data }: NodeProps<SourceFlowNode>) {
  const d = (data ?? { variant: 'number', value: 0 }) as SourceFlowNodeData;
  const subtitle = d.variant === 'number' ? d.visionSubtitle : undefined;
  const { registerPortKind, unregisterPortKinds } = useNode();

  // Determine what kind of data this source produces
  const produces: HandleKind = d.variant === 'number' ? 'rational' : 'cpa';

  // Register the port kind when the component mounts or produces changes
  useEffect(() => {
    registerPortKind(id, 'out', { produces });
    return () => unregisterPortKinds(id);
  }, [id, produces, registerPortKind, unregisterPortKinds]);

  return (
    <div className="relative h-80 w-52 -translate-x-[30%] -translate-y-[50%]">
      <TrackIdBadge trackId={readTrackId(d)} />
      <FlowNodeCard
        family="input"
        title={sourceTitle(d)}
        content={<span className="text-xs font-black text-slate-100">{sourceMain(d)}</span>}
        subtitle={subtitle}
      />
      <ClickableHandle
        type="source"
        position={Position.Right}
        id="out"
        nodeId={id}
        produces={produces}
        style={{ transform: 'translateX(100px)' }}
      />
    </div>
  );
}
