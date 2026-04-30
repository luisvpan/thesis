import type { OperatorType } from '@/types/card-types';
import { motion } from 'motion/react';


interface OperatorCardProps {
  operator: OperatorType;
  isDraggable?: boolean;
  onClick?: () => void;
  size?: 'small' | 'medium' | 'large';
}

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
