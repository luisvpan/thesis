import type { OperatorType } from '@/types/card-types';
import { motion } from 'motion/react';


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


const sizeClasses = {
  small: 'min-w-[5.5rem] min-h-[4.5rem] px-2 py-2',
  medium: 'min-w-[7rem] min-h-[5.5rem] px-3 py-3',
  large: 'min-w-[9rem] min-h-[7rem] px-4 py-4',
};

const labelClasses = {
  small: 'text-sm',
  medium: 'text-base',
  large: 'text-lg',
};

export function OperatorCard({ operator, isDraggable = false, onClick, size = 'medium' }: OperatorCardProps) {


  return (
    <motion.div
      drag={isDraggable}
      dragMomentum={false}
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className={isDraggable ? 'cursor-grab active:cursor-grabbing' : onClick ? 'cursor-pointer' : ''}
    >
      <div
        className={`${sizeClasses[size]} rounded-lg flex items-center justify-center border-2 border-[#ef4444] bg-transparent`}
      >
        <span className={`${labelClasses[size]} font-semibold text-center text-blue-400`}>{operator}</span>
      </div>
    </motion.div>
  );
}
