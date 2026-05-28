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
      return 'Palito';
    case 'food':
      return 'Comida';
    case 'criteria':
      if (data.properties.includes('size')) return 'Tamaño';
      if (data.properties.includes('color')) return 'Color';
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
      return data.color ?? 'palito';
    case 'food':
      return data.food;
    case 'criteria':
      if (data.values.size) return data.values.size;
      if (data.values.color) return data.values.color;
      return data.yoloClass;
  }
}
