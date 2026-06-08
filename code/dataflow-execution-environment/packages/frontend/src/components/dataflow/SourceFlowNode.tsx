import { useEffect } from 'react';
import type { Node, NodeProps } from '@xyflow/react';
import { Position } from '@xyflow/react';
import { ClickableHandle } from './ClickableHandle';
import type { ShapeType, ShapeSize, ShapeColor, FoodType, MontessoriColor, CapColor, StickColor } from '@/types/card-types';
import type { CriteriaProperty, CriteriaValues } from '@/data/yoloDeckCatalog';
import { TrackIdBadge } from './TrackIdBadge';
import { readTrackId, type VisionNodeMeta } from '@/contexts/node/visionNodeMeta';
import { useNode } from '@/contexts/NodeContext';
import type { HandleKind } from './handle-kinds';
import { useFlowNodeShellClass } from './useFlowNodeShellClass';
import { SOURCE_NODE_WRAPPER_CLASS } from './source-flow/sourceNodeLayout';
import { renderSourceFlowNodeBody } from './source-flow/renderSourceFlowNodeBody';
import { isNumberMergeTail } from '@/utils/numberTouchMerge';

type VisionSynced = VisionNodeMeta;

export type SourceFlowNodeData = VisionSynced &
  (
    | {
        variant: 'number';
        /** Valor mostrado y enviado (puede ser multi-dígito si hay fusión táctil). */
        value: number;
        /** Dígito 0–9 de esta carta física antes de fusionar con vecinas. */
        digitValue?: number;
        /** ID del nodo primario del grupo táctil (solo si hay fusión). */
        numberMergePrimaryId?: string;
        /** ID del nodo final del grupo táctil (solo si hay fusión). */
        numberMergeTailId?: string;
        visionSubtitle?: string;
      }
    | { variant: 'shape'; yoloClass: string; shape: ShapeType; size: ShapeSize; color: ShapeColor }
    | { variant: 'food'; yoloClass: string; food: FoodType }
    | { variant: 'montessori'; yoloClass: string; color: MontessoriColor }
    | { variant: 'cap'; yoloClass: string; color: CapColor }
    | { variant: 'stick'; yoloClass: string; color?: StickColor }
    | { variant: 'criteria'; yoloClass: string; properties: CriteriaProperty[]; values: CriteriaValues }
    | {
        variant: 'dice';
        /** Valor obtenido tras lanzar (1–6). */
        value?: number;
        /** Cara mostrada durante la animación. */
        previewFace?: number;
        isRolling?: boolean;
      }
  );

export type SourceFlowNode = Node<SourceFlowNodeData, 'source'>;

export function SourceFlowNode({ id, data }: NodeProps<SourceFlowNode>) {
  const d = (data ?? { variant: 'number', value: 0 }) as SourceFlowNodeData;
  const { registerPortKind, unregisterPortKinds } = useNode();

  const produces: HandleKind =
    d.variant === 'number' || d.variant === 'dice'
      ? 'rational'
      : d.variant === 'criteria'
        ? 'keyword'
        : 'cpa';
  const wrapperClass = SOURCE_NODE_WRAPPER_CLASS[d.variant];
  const shellClass = useFlowNodeShellClass();

  const showOutHandle =
    (d.variant !== 'number' || isNumberMergeTail(id, d)) &&
    (d.variant !== 'dice' || d.value !== undefined);

  useEffect(() => {
    if (!showOutHandle) {
      unregisterPortKinds(id);
      return;
    }
    registerPortKind(id, 'out', { produces });
    return () => unregisterPortKinds(id);
  }, [id, produces, showOutHandle, registerPortKind, unregisterPortKinds]);

  return (
    <div className={`${wrapperClass} ${shellClass}`}>
      <TrackIdBadge trackId={readTrackId(d)} />
      {renderSourceFlowNodeBody(d, id)}
      {showOutHandle ? (
        <ClickableHandle
          type="source"
          position={Position.Right}
          id="out"
          nodeId={id}
          handleVariant="input-out"
          produces={produces}
          style={{ transform: 'translateX(100px)' }}
        />
      ) : null}
    </div>
  );
}
