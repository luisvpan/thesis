import { motion } from 'motion/react';
import type { MathOperatorType } from '@/types/card-types';

interface OperatorFlowCardProps {
  operator: MathOperatorType;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

/** Misma idea que NumberCard: caja fija + borde; rojo en lugar de azul; nombre arriba, símbolo dentro. */
const MATH_OPERATOR_FLOW: Record<MathOperatorType, { name: string; symbol: string }> = {
  adicion: { name: 'Adición', symbol: '+' },
  sustraccion: { name: 'Sustracción', symbol: '−' },
  multiplicacion: { name: 'Multiplicación', symbol: '×' },
  division: { name: 'División', symbol: '÷' },
};

const sizeClasses = {
  small:
    'box-border flex h-[4.5rem] w-[5.5rem] items-center justify-center rounded-lg border-2 border-red-500 bg-transparent',
  medium:
    'box-border flex h-[5.5rem] w-[7rem] items-center justify-center rounded-lg border-2 border-red-500 bg-transparent',
  large:
    'box-border flex h-[7rem] w-[9rem] items-center justify-center rounded-lg border-2 border-red-500 bg-transparent',
};

const nameClasses = {
  small: 'text-[11px] font-semibold text-red-400',
  medium: 'text-xs font-semibold text-red-400',
  large: 'text-sm font-semibold text-red-400',
};

const symbolClasses = {
  small: 'text-2xl font-bold text-red-400',
  medium: 'text-3xl font-bold text-red-400',
  large: 'text-4xl font-bold text-red-400',
};

export function OperatorFlowCard({
  operator,
  isDraggable = false,
  onClick,
  size = 'medium',
}: OperatorFlowCardProps) {
  const { name, symbol } = MATH_OPERATOR_FLOW[operator];

  return (
    <motion.div
      drag={isDraggable}
      dragMomentum={false}
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      onClick={onClick}
      className={
        isDraggable ? 'cursor-grab active:cursor-grabbing' : onClick ? 'cursor-pointer' : ''
      }
    >
      <div className="flex flex-col items-center gap-1">
        <span className={`${nameClasses[size]} text-center leading-tight`}>{name}</span>
        <div className={sizeClasses[size]}>
          <span className={`${symbolClasses[size]} text-center`}>{symbol}</span>
        </div>
      </div>
    </motion.div>
  );
}
