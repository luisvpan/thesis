import type { SourceFlowNodeData } from '../SourceFlowNode';
import { isPictorialColorYoloClass } from '@/data/pictorialColors';

/** Cartas pictóricas de tamaño sin forma explícita (distintas de sm_/md_/lg_* + figura). */
const PICTORIAL_SIZE_ONLY_YOLO = new Set(['small', 'medium', 'large']);

export function sourceTitle(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return 'Numero';
  if (data.variant === 'shape' && isPictorialColorYoloClass(data.yoloClass)) return 'Color';
  if (data.variant === 'shape' && PICTORIAL_SIZE_ONLY_YOLO.has(data.yoloClass)) return 'Tamaño';
  if (data.variant === 'shape') return 'Forma';
  if (data.variant === 'montessori') return 'Cubo';
  if (data.variant === 'cap') return 'Tapa';
  if (data.variant === 'stick') return 'Palito';
  return 'Comida';
}

export function sourceMain(data: SourceFlowNodeData): string {
  if (data.variant === 'number') return String(data.value);
  if (data.variant === 'shape' && isPictorialColorYoloClass(data.yoloClass)) return data.color;
  if (data.variant === 'shape' && PICTORIAL_SIZE_ONLY_YOLO.has(data.yoloClass)) return data.size;
  if (data.variant === 'shape') return `${data.shape} ${data.size}`;
  if (data.variant === 'montessori') return data.color;
  if (data.variant === 'cap') return data.color;
  if (data.variant === 'stick') return data.color;
  return data.food;
}
