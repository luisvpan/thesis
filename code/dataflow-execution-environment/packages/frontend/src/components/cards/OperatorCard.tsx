import { CardBase } from './CardBase';
import type { OperatorType } from '@/types/card-types';

interface OperatorCardProps {
  operator: OperatorType;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

interface OperatorConfig {
  symbol: string;
  name: string;
}

const operatorConfig: Record<OperatorType, OperatorConfig> = {
  // Operadores matemáticos básicos
  'adicion': {
    symbol: '+',
    name: 'Adición',
  },
  'sustraccion': {
    symbol: '−',
    name: 'Sustracción',
  },
  'multiplicacion': {
    symbol: '×',
    name: 'Multiplicación',
  },
  'division': {
    symbol: '÷',
    name: 'División',
  },

  // Operadores de orden y comparación
  'orden-mayor-menor': {
    symbol: '9→1',
    name: 'Orden ▼',
  },
  'orden-menor-mayor': {
    symbol: '1→9',
    name: 'Orden ▲',
  },
  'comparar-figuras': {
    symbol: '🔺=',
    name: 'Comparar Figuras',
  },
  'comparar-carros': {
    symbol: '🚗=',
    name: 'Comparar Carros',
  },
  'comparar-comidas': {
    symbol: '🍎=',
    name: 'Comparar Comidas',
  },
  'comparar-animales': {
    symbol: '🐾=',
    name: 'Comparar Animales',
  },
  'comparar-personas': {
    symbol: '👥=',
    name: 'Comparar Personas',
  },

  // Operadores de filtrado
  'filtrar-general': {
    symbol: '⊲',
    name: 'Filtrar',
  },
  'filtrar-figuras': {
    symbol: '⊲🔺',
    name: 'Filtrar Figuras',
  },
  'filtrar-carros': {
    symbol: '⊲🚗',
    name: 'Filtrar Carros',
  },
  'filtrar-comidas': {
    symbol: '⊲🍎',
    name: 'Filtrar Comidas',
  },
  'filtrar-animales': {
    symbol: '⊲🐾',
    name: 'Filtrar Animales',
  },
  'filtrar-personas': {
    symbol: '⊲👥',
    name: 'Filtrar Personas',
  },

  // Operadores de conjuntos
  'union': {
    symbol: '∪',
    name: 'Unión',
  },
  'interseccion': {
    symbol: '∩',
    name: 'Intersección',
  },
  'diferencia': {
    symbol: '−',
    name: 'Diferencia',
  },
  'complemento': {
    symbol: '∁',
    name: 'Complemento',
  },
};

export function OperatorCard({ operator, isDraggable = false, onClick, size = 'medium' }: OperatorCardProps) {
  const config = operatorConfig[operator];

  return (
    <CardBase 
      borderColor="#ef4444" // Rojo
      isDraggable={isDraggable}
      onClick={onClick}
      size={size}
      cardType="OPERADOR"
      cardName={config.name}
    >
      {config.symbol}
    </CardBase>
  );
}
