import type { ShapeColor } from '@/types/card-types';

/** Clases YOLO de cartas de color pictórico (círculo de color). */
export const PICTORIAL_COLOR_YOLO_CLASSES = [
  'green',
  'purple',
  'red',
  'orange',
  'yellow',
  'blue',
] as const;

export type PictorialColorYoloClass = (typeof PICTORIAL_COLOR_YOLO_CLASSES)[number];

export const YOLO_CLASS_TO_SHAPE_COLOR: Record<PictorialColorYoloClass, ShapeColor> = {
  green: 'verde',
  purple: 'morado',
  red: 'rojo',
  orange: 'naranja',
  yellow: 'amarillo',
  blue: 'azul',
};

export function isPictorialColorYoloClass(yoloClass: string): yoloClass is PictorialColorYoloClass {
  return (PICTORIAL_COLOR_YOLO_CLASSES as readonly string[]).includes(yoloClass);
}
