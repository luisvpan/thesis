import { resolveStickColor } from '@/types/card-types';
import type { SourceFlowNodeData } from '../SourceFlowNode';

export function sourceTitle(data: SourceFlowNodeData): string {
  switch (data.variant) {
    case 'number':
      return 'Numero';
    case 'shape':
      return 'Forma';
    case 'montessori':
      return 'Cubo';
    case 'cap':
      return 'Tapa';
    case 'stick':
      return 'Paleta';
    case 'food':
      return 'Comida';
    case 'criteria':
      if (data.properties.includes('size')) return 'Tamaño';
      if (data.properties.includes('color')) return 'Color';
      if (data.properties.includes('subtype')) return 'Forma';
      return 'Criterio';
  }
}

export function sourceMain(data: SourceFlowNodeData): string {
  switch (data.variant) {
    case 'number':
      return String(data.value);
    case 'shape':
      return `${data.shape} ${data.size}`;
    case 'montessori':
      return data.color;
    case 'cap':
      return data.color;
    case 'stick':
      return resolveStickColor(data.color, data.yoloClass);
    case 'food':
      return data.food;
    case 'criteria':
      if (data.values.size) return data.values.size;
      if (data.values.color) return data.values.color;
      if (data.values.subtype) return data.values.subtype;
      return data.yoloClass;
  }
}
